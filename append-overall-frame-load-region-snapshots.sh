
idx=0
for overall in overall-frame-load-region/frame-load-region_overall*.png
do
    echo $overall

    instant=${overall/_overall/_instant}
    pose=${instant/instant/instant_pose}
    savefname=${overall/_overall*/}
    savefname=${savefname}_$(printf %08d $idx).png
    pose_cut=${pose/pose/pose_cut}
    convert $pose -crop 600x600+250+500 -quality 10 -resize 200% $pose_cut
    convert +append $pose_cut $instant $overall $savefname

    idx=$(expr $idx + 1)
done

echo "now converting to mp4"
# ffmpeg -framerate 5 -i overall-frame-load-region/frame-load-region_%08d.png -vcodec libx264 -pix_fmt yuv420p -r 30 overall-frame-load-region/frame-load-region.mp4
ffmpeg -framerate 60 -i overall-frame-load-region/frame-load-region_%08d.png -vcodec libx264 -pix_fmt yuv420p -r 60 overall-frame-load-region/frame-load-region.mp4
