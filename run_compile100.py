import sys
sys.path.insert(0, r"F:\\ai_agent\Corpus2Skill")
from pathlib import Path
from corpus2skill.config import CompileConfig
from corpus2skill.compile import compile_corpus

print("compile start: _c2s_sample100 -> _c2s_sample100_out")
compile_corpus(CompileConfig(
    input_dir=Path(r"F:\\ai_agent\_c2s_sample100"),
    output_dir=Path(r"F:\\ai_agent\_c2s_sample100_out"),
))
print("compile DONE")