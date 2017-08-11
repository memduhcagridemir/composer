#!/usr/bin/env bash

# configure locale
sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
sed -i -e 's/# tr_TR.UTF-8 UTF-8/tr_TR.UTF-8 UTF-8/' /etc/locale.gen
echo 'LANG="en_US.UTF-8"'>/etc/default/locale
dpkg-reconfigure --frontend=noninteractive locales
update-locale LANG=en_US.UTF-8

# configure timezone
ln -fs /usr/share/zoneinfo/Europe/Istanbul /etc/localtime
dpkg-reconfigure -f noninteractive tzdata

apt-get update
apt-get -y upgrade

# install python2 dev tools
apt-get install -y python2.7-dev python-pip

# install tensorflow
pip install tensorflow

# install tools for conversions
apt-get install -y abcmidi timidity lame
