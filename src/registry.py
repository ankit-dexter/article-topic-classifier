import os
import json
import boto3
import logging
from pathlib import Path
from datetime import datetime, timezone


class ModelRegistry:
    def __init__(self, registry_bucket: str | None, logger: logging.Logger):
        self.registry_bucket = registry_bucket
        self.logger = logger
        self.s3 = boto3.client("s3") if registry_bucket else None

    # =====================================================
    # MODE DETECTION
    # =====================================================
    @property
    def is_cloud(self) -> bool:
        return self.registry_bucket is not None

    # =====================================================
    # LOAD EXISTING MODEL (LOCAL OR CLOUD)
    # =====================================================
    def load_existing_model_path(self, local_output_dir: str) -> str | None:
        if self.is_cloud:
            return self._load_from_s3()
        else:
            return self._load_local(local_output_dir)

    # ----------------- LOCAL -----------------------------
    def _load_local(self, local_output_dir: str) -> str | None:
        self.logger.info("[REGISTRY][LOCAL] Checking for existing model")

        if not os.path.exists(local_output_dir):
            self.logger.info("[REGISTRY][LOCAL] Directory does not exist")
            return None

        required_files = ["config.json"]
        if not all(
            os.path.exists(os.path.join(local_output_dir, f))
            for f in required_files
        ):
            self.logger.info("[REGISTRY][LOCAL] Required files missing")
            return None

        self.logger.info(
            f"[REGISTRY][LOCAL] Found existing model at {local_output_dir}"
        )
        return local_output_dir

    # ----------------- CLOUD -----------------------------
    def _load_from_s3(self) -> str | None:
        self.logger.info("[REGISTRY][CLOUD] Checking latest.json")

        try:
            obj = self.s3.get_object(
                Bucket=self.registry_bucket,
                Key="pointers/latest.json",
            )
        except self.s3.exceptions.NoSuchKey:
            self.logger.info(
                "[REGISTRY][CLOUD] latest.json not found → training from scratch"
            )
            return None

        latest = json.loads(obj["Body"].read())
        current_version = latest.get("current_version")

        if not current_version:
            self.logger.info(
                "[REGISTRY][CLOUD] current_version is null → training from scratch"
            )
            return None

        prefix = f"models/{current_version}/"

        response = self.s3.list_objects_v2(
            Bucket=self.registry_bucket,
            Prefix=prefix,
            MaxKeys=1,
        )

        if "Contents" not in response:
            self.logger.warning(
                f"[REGISTRY][CLOUD] Version {current_version} missing in S3 → fallback to scratch"
            )
            return None

        local_dir = f"/tmp/{current_version}"
        os.makedirs(local_dir, exist_ok=True)

        self.logger.info(
            f"[REGISTRY][CLOUD] Downloading model version {current_version}"
        )

        full_response = self.s3.list_objects_v2(
            Bucket=self.registry_bucket,
            Prefix=prefix,
        )

        for obj in full_response.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            filename = key.split("/")[-1]
            local_path = os.path.join(local_dir, filename)
            self.s3.download_file(
                self.registry_bucket,
                key,
                local_path,
            )

        self.logger.info(
            f"[REGISTRY][CLOUD] Model downloaded to {local_dir}"
        )
        return local_dir

    # =====================================================
    # SAVE NEW VERSION (CLOUD ONLY)
    # =====================================================

    def save_new_version(self, local_model_dir: str):
        if not self.is_cloud:
            self.logger.info("[REGISTRY][LOCAL] Skipping S3 upload")
            return

        self.logger.info("[REGISTRY][CLOUD] Uploading new model version")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_name = f"v{timestamp}"
        prefix = f"models/{version_name}/"

        # -------------------------
        # Upload model artifacts
        # -------------------------
        for root, _, files in os.walk(local_model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_model_dir)
                s3_key = f"{prefix}{relative_path}"

                self.s3.upload_file(
                    local_path,
                    self.registry_bucket,
                    s3_key,
                )

        self.logger.info(f"[REGISTRY][CLOUD] Upload complete: {version_name}")

        # -------------------------
        # Verify upload
        # -------------------------
        verify = self.s3.list_objects_v2(
            Bucket=self.registry_bucket,
            Prefix=prefix,
            MaxKeys=1,
        )

        if "Contents" not in verify:
            raise RuntimeError(
                f"[REGISTRY][CLOUD] Upload verification failed for {version_name}"
            )

        # -------------------------
        # Update metadata.json
        # -------------------------
        metadata_key = "pointers/metadata.json"

        try:
            obj = self.s3.get_object(
                Bucket=self.registry_bucket,
                Key=metadata_key,
            )
            metadata = json.loads(obj["Body"].read())
            self.logger.info("[REGISTRY][CLOUD] Existing metadata loaded")
        except Exception:
            self.logger.warning(
                "[REGISTRY][CLOUD] metadata.json missing — creating new"
            )
            metadata = {}

        now_iso = datetime.now(timezone.utc).isoformat()

        metadata["last_trained_at"] = now_iso
        metadata["current_version"] = version_name

        # If first ever training
        if not metadata.get("last_full_train_at"):
            metadata["last_full_train_at"] = now_iso

        metadata["total_versions"] = metadata.get("total_versions", 0) + 1
        metadata["training_mode"] = "incremental"

        self.s3.put_object(
            Bucket=self.registry_bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2).encode("utf-8"),
        )

        self.logger.info("[REGISTRY][CLOUD] metadata.json updated")

        # -------------------------
        # Update latest.json
        # -------------------------
        latest_content = json.dumps(
            {"current_version": version_name},
            indent=2,
        ).encode("utf-8")

        self.s3.put_object(
            Bucket=self.registry_bucket,
            Key="pointers/latest.json",
            Body=latest_content,
        )

        self.logger.info(
            f"[REGISTRY][CLOUD] latest.json updated → {version_name}"
        )

     