# Importing required modules
import os


class GCloudSync:


    def sync_to_gcloud(self, bucket, path, name):
        """
        Syncs folder contents with the gcloud content (upload)
        """

        command = f"gsutil cp {path}/{name} gs://{bucket}/"
        os.system(command)


    def sync_from_gcloud(self, bucket, name, destination):
        """
        Syncs gcloud content with the folder content (download)
        """

        command = f"gsutil cp gs://{bucket}/{name} '{destination}'"
        os.system(command)
