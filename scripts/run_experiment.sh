set -euo pipefail
#параметры по умолчанию
DATA_DIR=${DATA_DIR:-data/processed}
MODEL_OUT=${MODEL_OUT:-models/model.joblib}
N_ESTIMATORS=${N_ESTIMATORS:-100}
MAX_DEPTH=${MAX_DEPTH:-5}
LEARNING_RATE=${LEARNING_RATE:-0.01}
SEED=${SEED:-42}
TAG=""

usage() {
  cat <<EOF
Использование: $0 [опции]
  -n, --n-estimators N   число деревьев (сейчас: $N_ESTIMATORS)
  -o, --model-out DIR    место сохранения модели (сейчас: $MODEL_OUT)
  -m, --max-depth N      глубина дерева (сейчас: $MAX_DEPTH)
  -s, --seed N           random seed (сейчас: $SEED)
  -l, --learning-rate N  скорость обучения (сейчас: $LEARNING_RATE)
  -t, --tag NAME         человекочитаемая метка запуска
  -h, --help             справка
EOF
}
#разбор аргументов
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--n-estimators)  N_ESTIMATORS="$2"; shift 2 ;;
    -o|--model-out)     MODEL_OUT="$2";    shift 2 ;;
    -m|--max-depth)     MAX_DEPTH="$2";    shift 2 ;;
    -s|--seed)          SEED="$2";         shift 2 ;;
    -l|--learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    -t|--tag)           TAG="$2";          shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "неизвестный аргумент: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

log "запуск эксперимента $MODEL_OUT"
log "параметры: n_estimators=$N_ESTIMATORS max_depth=$MAX_DEPTH learning_rate=$LEARNING_RATE seed=$SEED"

#проверки перед стартом
[[ -f "$DATA_DIR/train.csv" ]] || { log "нет $DATA_DIR/train.csv"; exit 1; }
[[ -f "$DATA_DIR/test.csv"  ]] || { log "нет $DATA_DIR/test.csv";  exit 1; }

START=$(date +%s)
#пайплайн
{
  python src/train.py \
    --train "$DATA_DIR/train.csv" \
    --model-out "$MODEL_OUT" \
    --n-estimators "$N_ESTIMATORS" \
    --learning-rate "$LEARNING_RATE" \
    --max-depth "$MAX_DEPTH" \
    --seed "$SEED"

  python src/evaluate.py \
    --model "$MODEL_OUT" \
    --test "$DATA_DIR/test.csv" \
    --out "reports/metrics$(date +%Y%m%d_%H%M%S).json"
} 2>&1 | tee -a "reports/train.log"

log "заняло $(( $(date +%s) - START )) секунд"