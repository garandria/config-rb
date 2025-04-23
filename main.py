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
    parser.add_argument("--configpath", type=str)
    parser.add_argument("--reproducible-check", type=str,
                        dest="configs_dir")
    parser.add_argument("--source", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--preset", type=str)

    args = parser.parse_args()

    n = args.n
    lz = f"0{len(str(n))}"
    version = args.version
    source = None
    archive = f"linux-{version}.tar.gz"
    if args.source is None:
        print(f"No source provided, checking for {archive}", flush=True)
        s = f"linux-{args.version}"
        if os.path.isdir(s):
            print(f"Source {s} already exist, cleaning...", flush=True, end=" ")
            build.distclean(s)
            print("done", flush=True)
            source = os.path.realpath(s)
        else:
            if not os.path.isfile(archive):
                print("Downloading source code archive...", flush=True)
                majorv = args.version.split('.', 1)[0]
                link = os.path.join("https://cdn.kernel.org/pub/linux/kernel/",
                                    f"v{majorv}.x", f"linux-{version}.tar.gz")
                archive = utils.download(link)
                print(f"Source tree extracted in {source}", flush=True)
            print(f"Extracting {archive}...", flush=True, end=" ")
            source = utils.extract(archive)
            print(f"-> {source}", flush=True)
    else:
        source = os.path.realpath(args.source)
        print(f"Source in {source}", flush=True)

    outdir = os.path.realpath(f"{args.output}-{args.version}")
    print(f"Output directory: {outdir}", flush=True)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print(f"{outdir} created", flush=True)

    env_list = [
        'KBUILD_BUILD_TIMESTAMP="Sun Jan 1 01:00:00 UTC 2023"',
        'KBUILD_BUILD_USER="user"',
        'KBUILD_BUILD_HOST="host"',
        'KBUILD_BUILD_VERSION="1"'
    ]
    preset = os.path.realpath("config.preset-x86_64")
    configintree = os.path.join(source, ".config")
    i = 1
    err = 0
    cset = set()
    if args.gen:
        confout = os.path.realpath(f"configs-{args.version}")
        print(f"Output directory: {confout}", flush=True)
        if not os.path.isdir(confout):
            os.mkdir(confout)
            print(f"{confout} created", flush=True)
        print(f"Generating {n} configurations...", flush=True)
        while i <= n:
            build.distclean(source)
            print(f"{i:{lz}}", end=" - ", flush=True)
            build.randconfig(source, preset)
            print(f"randconfig", end=" - ", flush=True)
            if utils.md5hash(configintree) in cset:
                continue
            else:
                cset.add(utils.md5hash(configintree))
            ok = build.build(source, source, env_list, "vmlinux",
                             nproc=args.threads, keep_metadata=args.debug)
            print(f"Build:", end=" ", flush=True)
            if ok:
                shutil.copy(configintree, os.path.join(confout, f"{i:{lz}}.config"))
                shutil.copy(os.path.join(source, "vmlinux"),
                            os.path.join(outdir, f"{i:{lz}}.vmlinux"))
                shutil.copy(os.path.join(source, "__time"),
                            os.path.join(outdir, f"{i:{lz}}.time"))
                i += 1
                print("Success", flush=True)
            else:
                err += 1
                print("Failure", flush=True)

        print(f"{i-1} configurations generated in {confout} and all build successfully.", flush=True)
        print(f"{err} configurations were not kept for they failed to build.", flush=True)

    configpath = os.path.realpath(args.configpath)
    if args.rebuild:
        for conf in map(lambda x: os.path.join(configpath, x), os.listdir(configpath)):
            build.distclean(source)
            print(f"{i:{lz}}", end=" - ", flush=True)
            shutil.copy(conf, configintree)
            ok = build.build(source, source, env_list, "vmlinux",
                             nproc=args.threads, keep_metadata=args.debug)
            print(f"Build:", end=" ", flush=True)
            if ok:
                shutil.copy(os.path.join(source, "vmlinux"),
                            os.path.join(outdir, f"{i:{lz}}.vmlinux"))
                shutil.copy(os.path.join(source, "__time"),
                            os.path.join(outdir, f"{i:{lz}}.time"))
                i += 1
                print("Success", flush=True)
            else:
                err += 1
                print("Failure", flush=True)


if __name__ == "__main__":
    main()
