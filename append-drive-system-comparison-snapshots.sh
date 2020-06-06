package_dir=${0%/*}

crop_region="620x620+200+0"
size="90%"
for region0 in $package_dir/drive-system-comparison/*system0_load-region_*.png
do
    echo $region0
    # echo $pose0
    # echo $region1
    # echo $pose1

    pose0=${region0/load-region/pose}
    pose0_cut=${pose0/pose/pose_cut}
    convert $pose0 -crop $crop_region -quality 10 -resize $size $pose0_cut
    savefname0=${region0/_load-region/}
    echo $savefname0
    convert $pose0_cut -gravity center $region0 -append $savefname0

    region1=${region0/system0/system1}
    pose1=${region1/load-region/pose}
    pose1_cut=${pose1/pose/pose_cut}
    convert $pose1 -crop $crop_region -quality 10 -resize $size $pose1_cut
    savefname1=${region1/_load-region/}
    echo $savefname1
    convert $pose1_cut -gravity center $region1 -append $savefname1

    region2=${region0/system0/system2}
    pose2=${region2/load-region/pose}
    pose2_cut=${pose2/pose/pose_cut}
    convert $pose2 -crop $crop_region -quality 10 -resize $size $pose2_cut
    savefname2=${region2/_load-region/}
    echo $savefname2
    convert $pose2_cut -gravity center $region2 -append $savefname2

    echo ""

done

echo "now converting to mp4"
for idx in $(seq 0 2)
do
    ffmpeg -y -framerate 5 -i $package_dir/drive-system-comparison/drive-system-comparison_system${idx}_%02d.png\
           -vcodec libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"\
           -r 30 $package_dir/drive-system-comparison/drive-system-comparison_system${idx}.mp4
done
