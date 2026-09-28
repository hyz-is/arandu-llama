package local_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/local"
)

// historicalRecipe is the q/v recipe that 25 durable checkpoints name by
// digest. It declares no adapter coverage, so it must keep that digest.
const historicalRecipe = `{"Method":"causal-sft-v1","BaseRevision":"489cb97981b8654bcfcf30ce1f94ed1b62e07b53","DataSHA256":"c40120ea427ee91ec50aa0c86418a9f0e547aae2a8cb2395ed31e5ff0ca9f77b","ExampleCount":103,"RequireLengthOrder":true,"Identity":{"IndexSHA256":"d5c7fee99574e9a05f901282aee04fc4fc3dccf094a659df48c6b8e9f39109c3","ConfigSHA256":"1f1b3751c38f16a63340df90a55e870bef0f0b2968d833825a605b7cf930a313","ReferenceSHA256":"69825f715b8f422e2e155be4bd2ed309e14d77ac9d45a65405a3b5886ca228dc","InitialAdapterSHA256":"75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c"},"Assembly":{"HeaderLimits":{"MaxHeaderBytes":16777216,"MaxTensors":65536,"MaxDimensions":32,"MaxMetadataEntries":65536,"MaxChunkBytes":4194304},"TensorCopyBytes":6102712320,"PersistentBytes":[11042374656,11042382848],"DeviceByLayer":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"EmbeddingDevice":0,"OutputDevice":1,"AdapterRank":4,"AdapterAlpha":8,"HashChunkBytes":4194304,"MaxInputElements":4399104,"MaxScoreElements":18455616,"MaxWorkingElements":1073741824,"Sequence":{"ChunkTokens":8,"MaxTokens":1074,"MaxOwnedElements":1073741824}},"MaxMPSBytes":26843545600,"Initializer":{"Seed":83,"PreludeBlocks":24,"PreludeWidth":32,"PreludeLow":0.01,"PreludeHigh":16,"Projections":[{"Name":"base_model.model.model.language_model.layers.3.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.3.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.7.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.7.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.11.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.11.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.15.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.15.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.19.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.19.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.23.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.23.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.27.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.27.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4},{"Name":"base_model.model.model.language_model.layers.31.self_attn.q_proj","Input":4096,"Output":8192,"Rank":4},{"Name":"base_model.model.model.language_model.layers.31.self_attn.v_proj","Input":4096,"Output":1024,"Rank":4}],"ExpectedSHA256":"75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c"},"Rotary":{"Theta":10000000,"Dimension":64,"MaxTokens":4096,"HalfPrecision":true,"ExpectedSHA256":"ec4437c15ead01576c3e6c7412c11daba363d17cee8bfce5c5d1c54e2e09d2d0"},"Optimizer":{"LearningRate":0.000002,"Beta1":0.9,"Beta2":0.999,"Epsilon":1E-8,"WeightDecay":0,"MaxGradientNorm":1},"LossScale":1,"MaxCheckpointBytes":299139072}`

const historicalRecipeSHA256 = "13cb2e2b507bf6c50f3a32b2276a09de051c36f48ca9a5764d181f3377305a5e"

func TestRecipeWithoutCoverageKeepsItsCheckpointDigest(t *testing.T) {
	var recipe local.Recipe
	if err := json.Unmarshal([]byte(historicalRecipe), &recipe); err != nil {
		t.Fatal(err)
	}
	if digest := recipe.Digest(); digest != historicalRecipeSHA256 {
		t.Fatalf("a recipe without coverage changed its digest: %s", digest)
	}
	encoded, err := json.Marshal(recipe)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("Coverage")) {
		t.Fatal("an undeclared coverage entered the recipe encoding")
	}
}

func TestARecipeWithoutDistillationEncodesAsBefore(t *testing.T) {
	var recipe local.Recipe
	if err := json.Unmarshal([]byte(historicalRecipe), &recipe); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(recipe)
	if err != nil {
		t.Fatal(err)
	}
	if recipe.Digest() != historicalRecipeSHA256 || bytes.Contains(encoded, []byte("Distillation")) {
		t.Fatalf("a recipe without distillation encodes differently:\n%s", encoded)
	}
}

// distilledRecipe is historicalRecipe with a distillation block. A reader that
// drops the block would train that recipe as plain SFT under the SFT digest.
func distilledRecipe() string {
	block := `,"Distillation":{"Method":"logit-kd-v1","DatasetID":"training","DatasetSHA256":"` + strings.Repeat("d", 64) + `","Student":{"name":"student","revision":"489cb97981b8654bcfcf30ce1f94ed1b62e07b53","weights_sha256":"` + strings.Repeat("1", 64) + `","tokenizer_sha256":"` + strings.Repeat("2", 64) + `","template_sha256":"` + strings.Repeat("3", 64) + `","runtime_sha256":"` + strings.Repeat("4", 64) + `","vocabulary":248320},"Teachers":[],"CacheLimits":{"MaxBytes":1,"MaxExamples":1,"MaxTokens":2,"MaxPositions":1,"MaxTopK":1,"MaxFeatureValues":0,"MaxMappingPairs":0}}}`
	return strings.TrimSuffix(historicalRecipe, "}") + block
}

func TestADeclaredDistillationEntersTheRecipeIdentity(t *testing.T) {
	var recipe local.Recipe
	if err := json.Unmarshal([]byte(distilledRecipe()), &recipe); err != nil {
		t.Fatal(err)
	}
	if recipe.Digest() == historicalRecipeSHA256 {
		t.Fatal("a declared distillation was dropped from the recipe identity")
	}
	encoded, err := json.Marshal(recipe)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(encoded, []byte(`"Distillation":{"Method":"logit-kd-v1"`)) {
		t.Fatalf("the distillation block did not survive the recipe encoding:\n%s", encoded)
	}
}
