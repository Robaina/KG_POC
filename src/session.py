import boto3
import gzip
import io


def get_assumed_session(
    role_arn, role_session_name, region_name, access_key_id, secret_access_key
):
    """
    Retrieves an assumed session using AWS STS to assume a role.

    Args:
        role_arn (str): The ARN of the role to assume.
        role_session_name (str): The name of the role session.
        region_name (str): The AWS region name.
        access_key_id (str): The AWS access key ID.
        secret_access_key (str): The AWS secret access key.

    Returns:
        boto3.Session: The assumed session.

    Example usage:
        session = get_assumed_session("arn:aws:iam::123456789012:role/MyRole", "MySession", "us-west-2", "ACCESS_KEY", "SECRET_KEY")
    """
    try:
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn, RoleSessionName=role_session_name
        )

        credentials = assumed_role_object["Credentials"]

        assumed_session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=region_name,
        )
        print("Assumed role successfully.")
        return assumed_session

    except Exception as e:
        print(f"Failed to assume role: {e}")
        return None


def read_first_n_lines(session, bucket_name, file_key, n=10):
    # Create S3 client using the provided session
    s3 = session.client("s3")

    # Get the object from S3
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)

    # Read the content of the file
    with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as gzipfile:
        lines = []
        for _ in range(n):
            line = gzipfile.readline().decode("utf-8").strip()
            if not line:
                break
            lines.append(line)

    return lines
