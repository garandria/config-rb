import os
import requests
import tarfile


def download(version):
    major, minor = version.split('.', 1)
    base_link = f"https://cdn.kernel.org/pub/linux/kernel/v{major}.x/"
    down_link = f"{base_link}linux-{version}.tar.gz"
    response = requests.get(down_link)
    if response.status_code == 200:
        with open(f"linux-{version}.tar.gz", "wb") as file:
            file.write(response.content)
        print(f"Downloaded linux-{version}.tar.gz successfully.")
    else:
        print(f"Failed to download linux-{version}.tar.gz. Status code: {response.status_code}")


def extract(archive):
    output = ""
    with tarfile.open(archive, 'r') as tar:
        output = tar.getnames()[0]
        tar.extractall()
    return output


