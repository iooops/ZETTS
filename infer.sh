
# !/usr/bin/env bash

# declare -a SPEAKERS=("/home/xingxing/COMP5214/dataset/esd/0018/Neutral/test/0018_000021.wav" "/home/xingxing/COMP5214/dataset/esd/0003/Neutral/test/0003_000021.wav" "/home/xingxing/COMP5214/dataset/esd/0010/Neutral/test/0010_000021.wav" "/home/xingxing/COMP5214/dataset/LibriTTS/train-clean-100/26/495/26_495_000004_000000.wav")
# declare -a FOLDERS=("ESD18" "ESD03" "ESD10" "LT26")
# declare -a EMOTIONS=("Angry" "Happy" "Neutral" "Sad" "Surprise" "Joyful" "Excited" "in a happy tone" "feeling down" "angry and sad" "a bit sad")

# for (( j=0; j<4; j++ ));
# do
#     for e in "${EMOTIONS[@]}"
#     do
#         python inference.py -f out/test3.txt -c logs/new_exp/grad_6323.pt -t 1000 -l 1 -s "${SPEAKERS[$j]}" -e "$e" -i 1 -fd "${FOLDERS[$j]}/${e}"
#     done
# done


# for i in $(seq 0 0.1 1);
# do
#     python inference.py -f out/test2.txt -c logs/new_exp/grad_6856.pt -t 1000 -l 1 -s /home/xingxing/COMP5214/dataset/esd/0018/Neutral/test/0018_000021.wav -e Happy -i "$i" -fd ESD18_happy/${i}
# done


python inference.py -f out/test3.txt -c checkpts/grad_6856.pt -t 1000 -l 1 -s ref_audio/esd0018_000021.wav -e Neutral -i 1 -fd LJ01_neutral_10
