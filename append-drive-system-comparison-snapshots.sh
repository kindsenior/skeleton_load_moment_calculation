
for region0 in drive-system-comparison/*system0_load-region_*.png
do
    echo $region0
    # echo $pose0
    # echo $region1
    # echo $pose1

    region1=${region0/system0/system1}
    region2=${region0/system0/system2}
    savefname=${region0/_system0_load-region/}
    echo $savefname
    # convert $region1 -gravity east $region0 +append $savefname
    convert +append $region0 $region1 $region2  $savefname

done

echo "now converting to mp4"
ffmpeg -framerate 5 -i drive-system-comparison/drive-system-comparison_%02d.png -vcodec libx264 -pix_fmt yuv420p -r 30 drive-system-comparison/drive-system-comparison.mp4
