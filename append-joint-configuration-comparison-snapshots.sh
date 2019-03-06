
idx=0
for region0 in joint-configuration-comparison/*configuration0_load-region_*.png
do
    echo $region0
    # echo $pose0
    # echo $region1
    # echo $pose1

    pose0=${region0/load-region/pose}
    pose0_cut=${pose0/pose/pose_cut}
    convert $pose0 -crop 500x500+400+500 -quality 10 -resize 200% $pose0_cut
    savefname0=${region0/_load-region/}
    echo $savefname0
    convert $pose0_cut -gravity east $region0 +append $savefname0

    region1=${region0/configuration0/configuration1}
    pose1=${region1/load-region/pose}
    pose1_cut=${pose1/pose/pose_cut}
    convert $pose1 -crop 500x500+400+500 -quality 10 -resize 200% $pose1_cut
    savefname1=${region1/_load-region/}
    echo $savefname1
    convert $pose1_cut -gravity east $region1 +append $savefname1
    echo ""

    idx=$(expr $idx + 1)
done

echo "now converting to mp4"
for idx in seq 0 1
do
    ffmpeg -framerate 5 -i joint-configuration-comparison/joint-configuration-comparison_configuration${idx}_%02d.png -vcodec libx264 -pix_fmt yuv420p -r 30 joint-configuration-comparison/joint-configuration-comparison_configuration${idx}.mp4
done
