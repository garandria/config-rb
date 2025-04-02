import os
import requests
import tarfile


def download(link):
    output = os.path.basename(link)
    response = requests.get(link)
    if response.status_code == 200:
        with open(output, "wb") as stream:
            stream.write(response.content)
        print(f"{output} downloaded successfully.")
        return output
    else:
        print(f"Failed to download {output}."\
              f"Status code: {response.status_code}")


def extract(archive):
    output = ""
    with tarfile.open(archive, 'r') as tar:
        output = tar.getnames()[0]
        tar.extractall()
    return output


