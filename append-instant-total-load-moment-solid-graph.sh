# convert -delay 20 -loop 0 *.png total-skeleton-load-moment-solid.gif

idx=0
for fname in instant-skeleton-load-moment-solid/*.png
do
    echo $fname
    echo ${fname//instant/total}
    savefname=${fname//instant-/}
    savefname=${savefname//\-[0-9]/}
    savefname=${savefname/.png/_$(printf %03d $idx).png}
    echo $savefname
    convert $fname -gravity east -splice 500x0 ${fname//instant/total} +append $savefname
    idx=$(expr $idx + 1)
done

# echo "now converting to gif..."
# convert -delay 20 -loop 0 skeleton-load-moment-solid/*.png skeleton-load-moment-solid/skeleton-load-moment-solid.gif

echo "now converting to mp4"
ffmpeg -framerate 5 -i skeleton-load-moment-solid/skeleton-load-moment-solid_%03d.png -vcodec libx264 -pix_fmt yuv420p -r 30 skeleton-load-moment-solid/skeleton-load-moment-solid.mp4
