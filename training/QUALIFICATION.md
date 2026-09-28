# Native training calculations

The Go packages provide bounded tensor calculations, decoder assembly, causal
SFT updates, teacher-cache contracts and constrained candidate updates. Model
identities, corpus revisions, numerical recipes and deployment placement belong
to explicit caller configuration. They are not default settings of this module.

`training/decoder` supports gated grouped-query attention and recurrent decoder
layers in the admitted storage format. Unsupported operations are refused;
accepting dimensions does not qualify an arbitrary architecture. Reference
hashes bind loading, initialization and rotary frequencies independently.

`training/local` is a bounded causal SFT executor. It preserves complete FP32
optimizer state and validates checkpoint payloads. New checkpoints bind the
entire numerical recipe. Historical checkpoints need caller-admitted manifest
hashes before they can be resumed. This executor does not implement the complete
multi-teacher pipeline or establish a quality improvement.

`training/local` can blend into each step the teacher-forced top-k
distributions of teachers that share the student's measured vocabulary. CPU
fixtures check two things: the blend against an independent trajectory, and
alpha zero against SFT bit for bit. They do not qualify teacher caches, model
quality or a real run.

Native primitives use the optional LibTorch bridge. No Python interpreter is
required. Builds without the native backend can inspect persisted state but
refuse tensor execution. The caller supplies compatible headers and libraries
and must qualify the actual runtime, devices, memory limits and numerical mode.

CPU fixtures check derivatives, ordered state updates, checkpoint integrity,
identity rejection and cancellation. Such checks do not qualify model quality,
GPU placement, distributed execution, quantization or a complete training run.
Project-specific scientific evidence and source-model lineage remain with the
recipe owner; package publication is not scientific promotion.
