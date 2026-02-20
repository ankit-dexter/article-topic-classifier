import os
import json
import boto3
import logging
from datetime import datetime, timezone


class ModelRegistry:
    """
    Model Registry abstraction.

    Cloud mode (registry_bucket provided):
      - reads pointers/latest.json
      - downloads model from models/<version>/
      - uploads new version to models/<new_version>/
      - updates pointers/metadata.json and pointers/latest.json

    Local mode (registry_bucket None):
      - checks local_output_dir for an existing model
      - no uploads
    """

    def __init__(self, registry_bucket: str | None, logger: logging.Logger):
        self.registry_bucket = registry_bucket
        self.logger = logger
        self.s3 = boto3.client("s3") if registry_bucket else None

    @property
    def is_cloud(self) -> bool:
        return self.registry_bucket is not None

    # =====================================================
    # LOAD EXISTING MODEL (LOCAL OR CLOUD)
    # =====================================================
    def load_existing_model_path(self, local_output_dir: str) -> str | None:
        if self.is_cloud:
            return self._load_from_s3()
        return self._load_local(local_output_dir)

    # ----------------- LOCAL -----------------------------
    def _load_local(self, local_output_dir: str) -> str | None:
        self.logger.info("[REGISTRY][LOCAL] Checking for existing model")
        if not os.path.exists(local_output_dir):
            self.logger.info("[REGISTRY][LOCAL] Directory does not exist")
            return None

        # Minimal integrity check for HF saved model
        required = ["config.json"]
        if not all(os.path.exists(os.path.join(local_output_dir, f)) for f in required):
            self.logger.info("[REGISTRY][LOCAL] Required files missing (config.json)")
            return None

        # Also ensure directory isn't empty
        if not os.listdir(local_output_dir):
            self.logger.info("[REGISTRY][LOCAL] Directory empty")
            return None

        self.logger.info(f"[REGISTRY][LOCAL] Found existing model at {local_output_dir}")
        return local_output_dir

    # ----------------- CLOUD -----------------------------
    def _load_from_s3(self) -> str | None:
        assert self.s3 is not None

        self.logger.info("[REGISTRY][CLOUD] Checking pointers/latest.json")

        # Read latest.json
        try:
            latest_obj = self.s3.get_object(
                Bucket=self.registry_bucket,
                Key="pointers/latest.json",
            )
            latest = json.loads(latest_obj["Body"].read())
        except Exception as e:
            self.logger.warning(f"[REGISTRY][CLOUD] latest.json not available ({e}) → scratch")
            return None

        current_version = latest.get("current_version")

        if not current_version:
            self.logger.info("[REGISTRY][CLOUD] current_version is null → training from scratch")
            return None

        prefix = f"models/{current_version}/"

        # Check if prefix exists
        exists = self.s3.list_objects_v2(
            Bucket=self.registry_bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        if "Contents" not in exists:
            self.logger.warning(
                f"[REGISTRY][CLOUD] Version {current_version} missing in S3 ({prefix}) → scratch"
            )
            return None

        local_dir = f"/tmp/{current_version}"
        os.makedirs(local_dir, exist_ok=True)

        self.logger.info(f"[REGISTRY][CLOUD] Downloading model version {current_version} → {local_dir}")

        # Download all objects under prefix
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.registry_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                filename = key.split("/")[-1]
                local_path = os.path.join(local_dir, filename)
                self.s3.download_file(self.registry_bucket, key, local_path)

        # Minimal integrity check after download
        cfg_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(cfg_path):
            self.logger.warning(
                f"[REGISTRY][CLOUD] Downloaded model missing config.json → scratch fallback"
            )
            return None

        self.logger.info(f"[REGISTRY][CLOUD] Model ready at {local_dir}")
        return local_dir

    # =====================================================
    # SAVE NEW VERSION (CLOUD ONLY)
    # =====================================================
    def save_new_version(
        self,
        local_model_dir: str,
        accuracy=None,
        f1=None,
        training_time=None,
        dataset_size=None,
        training_mode: str = "incremental",
    ):
        """
        Uploads local_model_dir to S3 as a new version and updates:
          - pointers/metadata.json
          - pointers/latest.json

        Only called when promotion gating has passed.
        """
        if not self.is_cloud:
            self.logger.info("[REGISTRY][LOCAL] Skipping S3 upload")
            return

        assert self.s3 is not None

        # Create version name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_name = f"v{timestamp}"
        prefix = f"models/{version_name}/"

        self.logger.info(f"[REGISTRY][CLOUD] Promoting new version: {version_name}")
        self.logger.info(f"[REGISTRY][CLOUD] Upload prefix: s3://{self.registry_bucket}/{prefix}")

        # Upload artifacts
        uploaded_count = 0
        for root, _, files in os.walk(local_model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel = os.path.relpath(local_path, local_model_dir)
                s3_key = f"{prefix}{rel}"

                self.s3.upload_file(local_path, self.registry_bucket, s3_key)
                uploaded_count += 1

        self.logger.info(f"[REGISTRY][CLOUD] Upload complete. Files uploaded: {uploaded_count}")

        # Verify upload
        verify = self.s3.list_objects_v2(
            Bucket=self.registry_bucket,
            Prefix=prefix,
            MaxKeys=1,
        )
        if "Contents" not in verify:
            raise RuntimeError(f"[REGISTRY][CLOUD] Upload verification failed for {version_name}")

        # Load existing metadata (or initialize)
        metadata_key = "pointers/metadata.json"
        try:
            meta_obj = self.s3.get_object(Bucket=self.registry_bucket, Key=metadata_key)
            metadata = json.loads(meta_obj["Body"].read())
            self.logger.info("[REGISTRY][CLOUD] Loaded existing metadata.json")
        except Exception as e:
            self.logger.warning(f"[REGISTRY][CLOUD] metadata.json not found/invalid ({e}) → creating new")
            metadata = {}

        now_iso = datetime.now(timezone.utc).isoformat()

        # Update core pointers
        metadata["last_trained_at"] = now_iso
        metadata["current_version"] = version_name

        # First full train stamp (set only once)
        if not metadata.get("last_full_train_at"):
            metadata["last_full_train_at"] = now_iso

        # Version counter (best-effort)
        metadata["total_versions"] = int(metadata.get("total_versions", 0)) + 1
        metadata["training_mode"] = training_mode

        # Store evaluation metrics + stats
        if accuracy is not None or f1 is not None:
            metadata["metrics"] = {
                "accuracy": float(accuracy) if accuracy is not None else None,
                "f1": float(f1) if f1 is not None else None,
            }

        if training_time is not None:
            metadata["training_time_seconds"] = float(training_time)

        if dataset_size is not None:
            metadata["dataset_size"] = int(dataset_size)

        # Write metadata.json
        self.s3.put_object(
            Bucket=self.registry_bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2).encode("utf-8"),
        )
        self.logger.info("[REGISTRY][CLOUD] metadata.json updated")

        # Update latest.json (ONLY after successful upload + metadata update)
        latest_key = "pointers/latest.json"
        latest_body = json.dumps({"current_version": version_name}, indent=2).encode("utf-8")

        self.s3.put_object(
            Bucket=self.registry_bucket,
            Key=latest_key,
            Body=latest_body,
        )
        self.logger.info(f"[REGISTRY][CLOUD] latest.json updated → {version_name}")