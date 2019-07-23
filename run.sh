python deepdream.py --input_img ./data/balloon.jpeg \
                    --device cuda \
                    --epoch 8 \
                    --lr 1e-2 \
                    --experiment_name test_inception \
                    --num_octave 5\
                    --octave_ratio 1.4 \
                    --layer 9
