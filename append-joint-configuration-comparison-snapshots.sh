package_dir=${0%/*}

for idx in $(seq 0 1)
do
    pose=$package_dir/joint-configuration-comparison/joint-configuration-comparison_configuration${idx}_initial-pose.png
    pose_cut=${pose/pose/pose_cut}
    convert $pose -crop "200x500+360+0" -quality 10 -resize 100% $pose_cut
done

crop_region="700x700+200+0"
size="90%"
for region0 in $package_dir/joint-configuration-comparison/*configuration0_load-region_*.png
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

    region1=${region0/configuration0/configuration1}
    pose1=${region1/load-region/pose}
    pose1_cut=${pose1/pose/pose_cut}
    convert $pose1 -crop $crop_region -quality 10 -resize $size $pose1_cut
    savefname1=${region1/_load-region/}
    echo $savefname1
    convert $pose1_cut -gravity center $region1 -append $savefname1
    echo ""

done

echo "now converting to mp4"
for idx in $(seq 0 1)
do
    ffmpeg -y -framerate 5 -i $package_dir/joint-configuration-comparison/joint-configuration-comparison_configuration${idx}_%02d.png\
           -vcodec libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"\
           -r 30 $package_dir/joint-configuration-comparison/joint-configuration-comparison_configuration${idx}.mp4
done
