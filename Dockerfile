FROM tuxmake/x86_64_gcc-10:20230912

RUN apt-get -y update && apt-get install -y \
    time \
    curl \
    libpam0g-dev \
    libgmp-dev libmpfr-dev libmpc-dev locales

# https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/issues/207#issuecomment-557520951
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen

WORKDIR /home
