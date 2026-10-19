modelscope download --dataset Shanghai_AI_Laboratory/WanJuan1_dot_0 --include 'raw/nlp/ChinaNews-cn/*' --local_dir /share/project/wuhaiming/data/dataset/WanJuan1_dot_0/
modelscope download --dataset Shanghai_AI_Laboratory/WanJuan1_dot_0 --include 'raw/nlp/Exam-cn/*' --local_dir /share/project/wuhaiming/data/dataset/WanJuan1_dot_0/
modelscope download --dataset Shanghai_AI_Laboratory/WanJuan1_dot_0 --include 'raw/nlp/Law-cn/*' --local_dir /share/project/wuhaiming/data/dataset/WanJuan1_dot_0/
modelscope download --dataset Shanghai_AI_Laboratory/WanJuan1_dot_0 --include 'raw/nlp/Patent-cn/*' --local_dir /share/project/wuhaiming/data/dataset/WanJuan1_dot_0/

cd /share/project/wuhaiming/data/dataset/WanJuan1_dot_0/raw/nlp/CN/
find . -type f -name '*.jsonl.tar.gz' -exec sh -c '
for f do
    dir=$(dirname "$f")
    tar -xzf "$f" -C "$dir"
done
' sh {} +