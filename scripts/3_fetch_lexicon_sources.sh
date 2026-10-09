#!/usr/bin/env bash
# Download the lexical resources the bias group lexicon is mined from into
# data/lexicon/_raw/, pinned to a commit and checked against a sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/lexicon/_raw

# dest | url | sha256
FILES=(
  "bbq/vocabulary.csv|https://raw.githubusercontent.com/nyu-mll/BBQ/bea11bd97d79217245b5871acd247b9d6eb24598/templates/vocabulary.csv|c60c02768e0a3406d4367d5bfe84cd526458712bfcf208f7bb350ab4928b1dbd"
  "bbq/vocabulary_proper_names.csv|https://raw.githubusercontent.com/nyu-mll/BBQ/bea11bd97d79217245b5871acd247b9d6eb24598/templates/vocabulary_proper_names.csv|17009b2c9d086c19c25b5107aa4792712fdf5024270e62f40bd4818ecb2e9c5d"
  "bias_bench/bias_attribute_words.json|https://raw.githubusercontent.com/McGill-NLP/bias-bench/4bca39b0cddf6d5a81430116aed7729ad3e14eb0/data/bias_attribute_words.json|221f9f997b6eee90c4b5791c6f60771fb91bbbe3631ac07a010bd8b026e50dbc"
  "holistic_bias/descriptors.json|https://raw.githubusercontent.com/facebookresearch/ResponsibleNLP/0ec714eb084217f44cd9ac466d9e988c795302f9/holistic_bias/dataset/v1.1/descriptors.json|c1328f0edfdeefb76302d360a01878df3db55f18026f4ea1956b4a0f52daf943"
  "holistic_bias/nouns.json|https://raw.githubusercontent.com/facebookresearch/ResponsibleNLP/0ec714eb084217f44cd9ac466d9e988c795302f9/holistic_bias/dataset/v1.1/nouns.json|9bbe15bf01ea8827fd5d68a81ee2b05e04bd5f79478d9027efec0b4dbd232ee6"
  "czarnowska/age.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/age.csv|ab10c6ed910a2b048bd5085972bb8e77ebe202e23801a77653872a6befae20a3"
  "czarnowska/countries.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/countries.csv|a1d69603202b49945048782b581e47ed0b43c4624abc9b40a820627f1152075a"
  "czarnowska/disability.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/disability.csv|17d3c0b76ab654c61550675a818693dd9789e31e0d553d710f09b3979d13547b"
  "czarnowska/gender.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/gender.csv|3ba00c25312238d346ccdae6209e49aa044bf7ded7682354351ac338835b241b"
  "czarnowska/race.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/race.csv|bb9703c3bd5561b652685bca3246d3a094106a1f87ae85330fd32629261b7a06"
  "czarnowska/religion.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/religion.csv|ac1ed0698f17f5f1b3c0dabd8b83b565d78f531f302a988551abc1f34b98bcee"
  "czarnowska/sexuality.csv|https://raw.githubusercontent.com/amazon-science/generalized-fairness-metrics/4e24cf02101c86c960b2453406ab7c448d528753/terms/identity_terms/sexuality.csv|f9b3db0369ed2c1ef445bba0e52debe14ad0277d506fc912cd20b45251d12036"
  "crows_pairs/crows_pairs_anonymized.csv|https://raw.githubusercontent.com/nyu-mll/crows-pairs/8aaac11c485473159ec9328a65253a5be9a479dc/data/crows_pairs_anonymized.csv|dfb36986ce0502abbaf7055b9176da3d08d48e07df1251991b5dfbcbceab9d0c"
  "debiaswe/definitional_pairs.json|https://raw.githubusercontent.com/tolga-b/debiaswe/6bb03ee2a0d7214339e513ca71cd0a99069dc29c/data/definitional_pairs.json|f2387fdd4ae4d956083ee7594f1401b1ba112914d40a99bb61d481de73db9614"
  "debiaswe/equalize_pairs.json|https://raw.githubusercontent.com/tolga-b/debiaswe/6bb03ee2a0d7214339e513ca71cd0a99069dc29c/data/equalize_pairs.json|b6e7dda64461e7828f8d184a51216a87aefcfc2b726d3691aa15b7acc2a1994c"
  "gn_glove/male_word_file.txt|https://raw.githubusercontent.com/uclanlp/gn_glove/7e0c7cabb67875bef76b0d3026846e1c2029c860/wordlist/male_word_file.txt|d431679ce3ef4134647e22cb5fd89e8dbee3f04636f1c7cbae5f28a369acf60f"
  "gn_glove/female_word_file.txt|https://raw.githubusercontent.com/uclanlp/gn_glove/7e0c7cabb67875bef76b0d3026846e1c2029c860/wordlist/female_word_file.txt|5f0803f056de3fbc459589bce26272d3c5453112a3a625fb8ee99c0fbbed5b35"
)

for entry in "${FILES[@]}"; do
  IFS='|' read -r dest url sha <<< "${entry}"
  mkdir -p "${RAW}/$(dirname "${dest}")"
  [ -f "${RAW}/${dest}" ] || curl -sSfL -o "${RAW}/${dest}" "${url}"
  echo "${sha}  ${RAW}/${dest}" | sha256sum -c --quiet - || { echo "sha256 mismatch: ${dest}" >&2; exit 1; }
done
echo "[fetch_lexicon_sources] ${#FILES[@]} files in ${RAW}"
