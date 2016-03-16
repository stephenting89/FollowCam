#! /bin/sh

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin://bin:/opt/bin

. /lib/init/vars.sh
. /lib/lsb/init-functions

case "$1" in 
	start)
	python /home/pi/fydp/FollowCam/FollowCam_temp.py
	exit 0
	;;
	stop)
	*)
	echo "Done"
	exit 1
	;;
esac
