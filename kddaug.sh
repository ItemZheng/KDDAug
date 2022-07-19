# Generate data for vqa cpv2
echo "######## GENERATE DATA #########"
python generate_yesno.py
python generate_number.py
python generate_color.py
python generate_other.py

# divide into high quality and low quality
echo "######## DIVIDE INTO HIGH QUALITY AND LOW QUALITY DATA #########"
CUDA_VISIBLE_DEVICES=0 python divide.py --ratio 0.9999999999999

#### assign new label for low and high quality data
echo "######## ASSIGN NEW LABEL #########"
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name number --split high
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name number --split low
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name other --split high
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name other --split low
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name color --split high
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name color --split low
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name paraphrasing --split high
CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name paraphrasing --split low
#CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name yesno --split low
#CUDA_VISIBLE_DEVICES=0 python assign_new_answer.py --dataset cpv2 --name paired --split low