import os
import argparse
import shutil
import build
import utils

def main():
    parser = argparse.ArgumentParser(
        description="Configuration Reproducible Builds")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--generate-configs", action="store_true", dest="gen")
    parser.add_argument("--output", type=str, default="build")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--reproducible-check", type=str,
                        dest="configs_dir")
    parser.add_argument("--source", type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    n = args.n
    lz = f"0{len(str(n))}"
    version = args.version
    source = None
    if args.source:
        source = os.path.realpath(args.source)
        print(f"Source tree in {source}", flush=True)
    if not os.path.isfile(f"linux-{version}.tar.gz") or source is None:
        print("Downloading source code archive...", flush=True)
        majorv = args.version.split('.', 1)[0]
        link = os.path.join("https://cdn.kernel.org/pub/linux/kernel/",
                            f"v{majorv}.x", f"linux-{version}.tar.gz")
        archive = utils.download(link)
        source = utils.extract(archive)
        print(f"Source tree extracted in {source}", flush=True)

    outdir = os.path.realpath(args.output)
    print(f"Output directory: {outdir}", flush=True)
    if not os.path.isdir():
        os.mkdir(outdir)
        print(f"{outdir} created", flush=True)

    env_list = [
        'KBUILD_BUILD_TIMESTAMP="Sun Jan 1 01:00:00 UTC 2023"',
        'KBUILD_BUILD_USER="user"',
        'KBUILD_BUILD_HOST="host"',
        'KBUILD_BUILD_VERSION="1"'
    ]

    abs_config_path = os.path.join(source, ".config")
    i = 1
    err = 0
    cset = set()
    if args.gen:
        print(f"Generating {n} configurations...", flush=True)
        while i <= n:
            build.distclean(source)
            print(f"{i:{lz}}", end=" - ", flush=True)
            build.randconfig(source, "config.preset-x86_64")
            print(f"randconfig", end=" - ", flush=True)
            if utils.md5hash(abs_config_path) in cset:
                continue
            else:
                cset.add(utils.md5hash(abs_config_path))
            ok = build.build(source, source, env_list, "vmlinux",
                             nproc=args.threads, keep_metadata=args.debug)
            print(f"Build:", end=" ", flush=True)
            if ok:
                shutil.copy(abs_config_path, os.path.join(outdir, f"{i:{lz}}.config"))
                shutil.copy(os.path.join(source, "vmlinux"),
                            os.path.join(outdir, f"{i:{lz}}.vmlinux"))
                i += 1
                print("Success", flush=True)
            else:
                err += 1
                print("Failure", flush=True)

        print(f"{i} configurations generated in {outdir} and all build successfully.", flush=True)
        print(f"{err} configurations were not kept for they failed to build.", flush=True)


if __name__ == "__main__":
    main()
