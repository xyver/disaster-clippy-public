"""
Cloudflare R2 Storage

S3-compatible cloud storage for backup files (ZIM, HTML archives, etc.)
Includes credential validation and expiration warnings.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime, timedelta
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Try to import boto3 (S3 SDK)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Run: pip install boto3")


@dataclass
class R2Config:
    """R2 configuration from environment variables"""
    access_key_id: str
    secret_access_key: str
    endpoint_url: str
    bucket_name: str
    token_expires: Optional[datetime] = None  # When the API token expires

    @classmethod
    def from_env(cls, bucket_env_var: str = "R2_BUCKET_NAME",
                 default_bucket: str = "disaster-clippy-backups",
                 key_prefix: str = "") -> Optional['R2Config']:
        """
        Load config from environment variables.

        Args:
            bucket_env_var: Environment variable name for bucket
            default_bucket: Default bucket name if env var not set
            key_prefix: Prefix for credential env vars (e.g., "SUBMISSIONS_" for R2_SUBMISSIONS_ACCESS_KEY_ID)
        """
        # Build credential env var names with optional prefix
        if key_prefix:
            access_key_var = f"R2_{key_prefix}ACCESS_KEY_ID"
            secret_key_var = f"R2_{key_prefix}SECRET_ACCESS_KEY"
        else:
            access_key_var = "R2_ACCESS_KEY_ID"
            secret_key_var = "R2_SECRET_ACCESS_KEY"

        access_key = os.getenv(access_key_var, "")
        secret_key = os.getenv(secret_key_var, "")

        # Fall back to default credentials if prefixed ones not set
        if not access_key and key_prefix:
            access_key = os.getenv("R2_ACCESS_KEY_ID", "")
        if not secret_key and key_prefix:
            secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "")

        endpoint = os.getenv("R2_ENDPOINT_URL", "")
        bucket = os.getenv(bucket_env_var, default_bucket)

        # Parse token expiration date if set
        expires_str = os.getenv("R2_TOKEN_EXPIRES", "")
        token_expires = None
        if expires_str:
            try:
                token_expires = datetime.fromisoformat(expires_str)
            except ValueError:
                logger.warning(f"Invalid R2_TOKEN_EXPIRES format: {expires_str}")

        if not all([access_key, secret_key, endpoint]):
            return None

        return cls(
            access_key_id=access_key,
            secret_access_key=secret_key,
            endpoint_url=endpoint,
            bucket_name=bucket,
            token_expires=token_expires
        )


class R2StorageError(Exception):
    """Base exception for R2 storage errors"""
    pass


class R2CredentialsExpired(R2StorageError):
    """Raised when R2 API credentials have expired"""
    pass


class R2CredentialsInvalid(R2StorageError):
    """Raised when R2 API credentials are invalid"""
    pass


class R2Storage:
    """
    Cloudflare R2 storage client for backup files.

    Usage:
        storage = R2Storage()

        # Check if configured
        if storage.is_configured():
            # Upload a file
            storage.upload_file("path/to/local.zim", "backups/source.zim")

            # Download a file
            storage.download_file("backups/source.zim", "path/to/local.zim")

            # List files
            files = storage.list_files("backups/")

            # Get download URL
            url = storage.get_download_url("backups/source.zim")
    """

    def __init__(self, config: Optional[R2Config] = None):
        self.config = config or R2Config.from_env()
        self._client = None
        self._last_error: Optional[str] = None

    def is_configured(self) -> bool:
        """Check if R2 credentials are configured"""
        return self.config is not None and BOTO3_AVAILABLE

    def _check_expiration(self) -> None:
        """Check if token is expired or expiring soon"""
        if not self.config or not self.config.token_expires:
            return

        now = datetime.now()
        expires = self.config.token_expires

        if now >= expires:
            self._last_error = f"R2 API TOKEN EXPIRED on {expires.isoformat()}. Please update credentials!"
            logger.error("=" * 60)
            logger.error("R2 API TOKEN EXPIRED!")
            logger.error(f"Expiration date: {expires.isoformat()}")
            logger.error("Please create a new token at: https://dash.cloudflare.com")
            logger.error("Then update R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in .env")
            logger.error("=" * 60)
            raise R2CredentialsExpired(self._last_error)

        # Warn if expiring within 2 days
        if now >= expires - timedelta(days=2):
            days_left = (expires - now).days
            hours_left = int((expires - now).total_seconds() / 3600)
            self._last_error = f"R2 API token expires in {hours_left} hours! Update soon."
            logger.warning("=" * 60)
            logger.warning(f"R2 API TOKEN EXPIRING SOON!")
            logger.warning(f"Expires: {expires.isoformat()} ({hours_left} hours remaining)")
            logger.warning("Create a new token at: https://dash.cloudflare.com")
            logger.warning("=" * 60)

    def _get_client(self):
        """Get or create the S3 client"""
        if not self.is_configured():
            raise R2StorageError("R2 storage not configured. Set R2_* environment variables.")

        self._check_expiration()

        if self._client is None:
            self._client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name='auto'  # R2 uses 'auto' for region
            )
        return self._client

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the R2 connection and return status.
        Use this to verify credentials are working.
        """
        result = {
            "configured": self.is_configured(),
            "connected": False,
            "bucket_exists": False,
            "error": None,
            "token_status": "ok",
            "token_expires": None
        }

        if not self.is_configured():
            result["error"] = "R2 not configured. Set R2_* environment variables."
            return result

        # Check token expiration
        if self.config.token_expires:
            result["token_expires"] = self.config.token_expires.isoformat()
            now = datetime.now()
            if now >= self.config.token_expires:
                result["token_status"] = "expired"
                result["error"] = "API token has expired! Please update credentials."
                return result
            elif now >= self.config.token_expires - timedelta(days=2):
                hours_left = int((self.config.token_expires - now).total_seconds() / 3600)
                result["token_status"] = f"expiring_soon ({hours_left}h remaining)"

        try:
            client = self._get_client()

            # Try to list buckets to verify credentials
            client.head_bucket(Bucket=self.config.bucket_name)
            result["connected"] = True
            result["bucket_exists"] = True

        except NoCredentialsError:
            result["error"] = "Invalid credentials - check R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '403':
                result["error"] = "Access denied - credentials may be invalid or expired"
            elif error_code == '404':
                result["connected"] = True
                result["error"] = f"Bucket '{self.config.bucket_name}' not found. Create it first."
            else:
                result["error"] = f"R2 error: {str(e)}"
        except R2CredentialsExpired as e:
            result["error"] = str(e)
            result["token_status"] = "expired"
        except Exception as e:
            result["error"] = f"Connection error: {str(e)}"

        return result

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        """
        Upload a file to R2.

        Args:
            local_path: Path to local file
            remote_key: Key (path) in R2 bucket

        Returns:
            True if successful
        """
        try:
            client = self._get_client()
            client.upload_file(local_path, self.config.bucket_name, remote_key)
            logger.info(f"Uploaded {local_path} to r2://{self.config.bucket_name}/{remote_key}")
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            self._last_error = str(e)
            return False

    def upload_fileobj(self, file_obj: BinaryIO, remote_key: str) -> bool:
        """Upload a file-like object to R2"""
        try:
            client = self._get_client()
            client.upload_fileobj(file_obj, self.config.bucket_name, remote_key)
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            self._last_error = str(e)
            return False

    def download_file(self, remote_key: str, local_path: str) -> bool:
        """
        Download a file from R2.

        Args:
            remote_key: Key (path) in R2 bucket
            local_path: Path to save locally

        Returns:
            True if successful
        """
        try:
            client = self._get_client()

            # Ensure local directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            client.download_file(self.config.bucket_name, remote_key, local_path)
            logger.info(f"Downloaded r2://{self.config.bucket_name}/{remote_key} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            self._last_error = str(e)
            return False

    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List files in R2 bucket.

        Args:
            prefix: Filter by key prefix (folder path)

        Returns:
            List of file info dicts with 'key', 'size', 'last_modified'
        """
        try:
            client = self._get_client()
            response = client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )

            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'size_mb': round(obj['Size'] / (1024 * 1024), 2),
                    'last_modified': obj['LastModified'].isoformat()
                })

            return files
        except ClientError as e:
            logger.error(f"List failed: {e}")
            self._last_error = str(e)
            return []

    def file_exists(self, remote_key: str) -> bool:
        """Check if a file exists in R2"""
        try:
            client = self._get_client()
            client.head_object(Bucket=self.config.bucket_name, Key=remote_key)
            return True
        except ClientError:
            return False

    def get_file_info(self, remote_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a file in R2"""
        try:
            client = self._get_client()
            response = client.head_object(Bucket=self.config.bucket_name, Key=remote_key)
            return {
                'key': remote_key,
                'size': response['ContentLength'],
                'size_mb': round(response['ContentLength'] / (1024 * 1024), 2),
                'last_modified': response['LastModified'].isoformat(),
                'content_type': response.get('ContentType', 'application/octet-stream')
            }
        except ClientError:
            return None

    def get_download_url(self, remote_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a pre-signed download URL.

        Args:
            remote_key: Key (path) in R2 bucket
            expires_in: URL expiration time in seconds (default 1 hour)

        Returns:
            Pre-signed URL or None if failed
        """
        try:
            client = self._get_client()
            url = client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.config.bucket_name,
                    'Key': remote_key
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"URL generation failed: {e}")
            self._last_error = str(e)
            return None

    def delete_file(self, remote_key: str) -> bool:
        """Delete a file from R2"""
        try:
            client = self._get_client()
            client.delete_object(Bucket=self.config.bucket_name, Key=remote_key)
            logger.info(f"Deleted r2://{self.config.bucket_name}/{remote_key}")
            return True
        except ClientError as e:
            logger.error(f"Delete failed: {e}")
            self._last_error = str(e)
            return False

    def copy_file(self, source_key: str, dest_key: str) -> bool:
        """
        Copy a file within R2 (server-side copy, no download/upload needed).

        Args:
            source_key: Source key (path) in R2 bucket
            dest_key: Destination key (path) in R2 bucket

        Returns:
            True if successful
        """
        try:
            client = self._get_client()
            copy_source = {'Bucket': self.config.bucket_name, 'Key': source_key}
            client.copy_object(
                CopySource=copy_source,
                Bucket=self.config.bucket_name,
                Key=dest_key
            )
            logger.info(f"Copied r2://{self.config.bucket_name}/{source_key} to {dest_key}")
            return True
        except ClientError as e:
            logger.error(f"Copy failed: {e}")
            self._last_error = str(e)
            return False

    def move_file(self, source_key: str, dest_key: str) -> bool:
        """
        Move a file within R2 (copy then delete original).

        Args:
            source_key: Source key (path) in R2 bucket
            dest_key: Destination key (path) in R2 bucket

        Returns:
            True if successful
        """
        if self.copy_file(source_key, dest_key):
            return self.delete_file(source_key)
        return False

    def get_last_error(self) -> Optional[str]:
        """Get the last error message"""
        return self._last_error

    def copy_to_bucket(self, source_key: str, dest_bucket: str, dest_key: str) -> bool:
        """
        Copy a file to a different bucket (server-side copy, no download/upload needed).

        This is useful for moving approved submissions to the backups bucket
        without transferring data over the network.

        Args:
            source_key: Source key (path) in this bucket
            dest_bucket: Destination bucket name
            dest_key: Destination key (path) in destination bucket

        Returns:
            True if successful
        """
        try:
            client = self._get_client()
            copy_source = {'Bucket': self.config.bucket_name, 'Key': source_key}
            client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            logger.info(f"Copied r2://{self.config.bucket_name}/{source_key} to r2://{dest_bucket}/{dest_key}")
            return True
        except ClientError as e:
            logger.error(f"Cross-bucket copy failed: {e}")
            self._last_error = str(e)
            return False

    def move_to_bucket(self, source_key: str, dest_bucket: str, dest_key: str) -> bool:
        """
        Move a file to a different bucket (copy then delete original).

        Args:
            source_key: Source key (path) in this bucket
            dest_bucket: Destination bucket name
            dest_key: Destination key (path) in destination bucket

        Returns:
            True if successful
        """
        if self.copy_to_bucket(source_key, dest_bucket, dest_key):
            return self.delete_file(source_key)
        return False


# =============================================================================
# SINGLETON INSTANCES - Separate buckets for backups and submissions
# =============================================================================

# Backups bucket (read-only for Railway, full access for global admin)
_backups_storage: Optional[R2Storage] = None

# Submissions bucket (write for Railway, read/delete for global admin)
_submissions_storage: Optional[R2Storage] = None

# Legacy single-bucket storage (for backward compatibility)
_r2_storage: Optional[R2Storage] = None


def get_r2_storage() -> R2Storage:
    """
    Get or create the R2 storage singleton (legacy single-bucket mode).

    Uses R2_BUCKET_NAME environment variable.
    For new code, prefer get_backups_storage() or get_submissions_storage().
    """
    global _r2_storage
    if _r2_storage is None:
        _r2_storage = R2Storage()
    return _r2_storage


def get_backups_storage() -> R2Storage:
    """
    Get storage for the backups bucket (official curated content).

    Environment variable: R2_BACKUPS_BUCKET (defaults to R2_BUCKET_NAME for compatibility)

    This bucket contains:
    - Official curated backups
    - Read by Railway proxy for downloads
    - Write only by global admin
    """
    global _backups_storage
    if _backups_storage is None:
        # Try R2_BACKUPS_BUCKET first, fall back to R2_BUCKET_NAME for compatibility
        backups_bucket = os.getenv("R2_BACKUPS_BUCKET", "")
        if backups_bucket:
            config = R2Config.from_env("R2_BACKUPS_BUCKET", "disaster-clippy-backups")
        else:
            # Fallback to single bucket mode
            config = R2Config.from_env("R2_BUCKET_NAME", "disaster-clippy-backups")
        _backups_storage = R2Storage(config)
    return _backups_storage


def get_submissions_storage() -> R2Storage:
    """
    Get storage for the submissions bucket (user-submitted content).

    Environment variables:
    - R2_SUBMISSIONS_BUCKET_NAME: Bucket name
    - R2_SUBMISSIONS_ACCESS_KEY_ID: Access key (falls back to R2_ACCESS_KEY_ID)
    - R2_SUBMISSIONS_SECRET_ACCESS_KEY: Secret key (falls back to R2_SECRET_ACCESS_KEY)

    This bucket contains:
    - pending/ - New submissions awaiting review
    - approved/ - Approved submissions (moved to backups bucket)
    - rejected/ - Rejected submissions

    If R2_SUBMISSIONS_BUCKET_NAME is not set, falls back to backups bucket
    (single-bucket mode for backward compatibility).
    """
    global _submissions_storage
    if _submissions_storage is None:
        submissions_bucket = os.getenv("R2_SUBMISSIONS_BUCKET_NAME", "")
        if submissions_bucket:
            # Use separate credentials for submissions bucket
            config = R2Config.from_env(
                bucket_env_var="R2_SUBMISSIONS_BUCKET_NAME",
                default_bucket="disaster-clippy-submissions",
                key_prefix="SUBMISSIONS_"
            )
        else:
            # Fallback to single bucket mode - use same as backups
            config = R2Config.from_env("R2_BUCKET_NAME", "disaster-clippy-backups")
        _submissions_storage = R2Storage(config)
    return _submissions_storage


def approve_submission(submission_key: str, dest_source_id: str, dest_filename: str) -> bool:
    """
    Approve a submission by moving it from submissions bucket to backups bucket.

    This performs a server-side copy (data stays in R2, no network transfer).

    Args:
        submission_key: Key in submissions bucket (e.g., 'pending/abc123.json')
        dest_source_id: Destination source ID in backups (e.g., 'community-guides')
        dest_filename: Destination filename (e.g., 'guide-123.json')

    Returns:
        True if successful
    """
    submissions = get_submissions_storage()
    backups = get_backups_storage()

    # Server-side copy to backups bucket
    dest_key = f"backups/{dest_source_id}/{dest_filename}"

    if submissions.copy_to_bucket(submission_key, backups.config.bucket_name, dest_key):
        # Move to approved folder in submissions bucket (keep record)
        approved_key = submission_key.replace("pending/", "approved/")
        submissions.move_file(submission_key, approved_key)
        logger.info(f"Approved submission: {submission_key} -> {dest_key}")
        return True
    return False


def reject_submission(submission_key: str, reason: str = "") -> bool:
    """
    Reject a submission by moving it to the rejected folder.

    Args:
        submission_key: Key in submissions bucket (e.g., 'pending/abc123.json')
        reason: Optional rejection reason

    Returns:
        True if successful
    """
    submissions = get_submissions_storage()

    rejected_key = submission_key.replace("pending/", "rejected/")
    if submissions.move_file(submission_key, rejected_key):
        logger.info(f"Rejected submission: {submission_key} -> {rejected_key}")
        return True
    return False


def reset_storage_singletons():
    """Reset all storage singletons (useful when config changes)"""
    global _r2_storage, _backups_storage, _submissions_storage
    _r2_storage = None
    _backups_storage = None
    _submissions_storage = None
