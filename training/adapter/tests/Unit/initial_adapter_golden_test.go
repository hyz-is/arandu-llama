package unit_test

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/adapter"
)

// initialGoldenSHA256 is the digest WriteInitialLoRA produced for
// initialGoldenTargets before the LoRA exporter shared its encoder. Any change
// to that encoder or to the initializer moves it.
const initialGoldenSHA256 = "cc587c8ce9410aff048cec7593096b860dc9f83d250b01191d28c37f276beb5d"

var initialGoldenTargets = []services.LoRATarget{
	{Name: "blk.0.attn_q.weight", Input: 12, Output: 20},
	{Name: "blk.0.ssm_out.weight", Input: 18, Output: 12},
	{Name: "blk.1.ffn_down.weight", Input: 7, Output: 12},
}

func TestInitialLoRABytesAreUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "golden.gguf")
	digest, err := services.WriteInitialLoRA(path, "qwen35", 4, 8, 59, initialGoldenTargets)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != digest {
		t.Fatalf("WriteInitialLoRA reported %s for a file whose SHA-256 is %s", digest, got)
	}
	if digest != initialGoldenSHA256 {
		t.Fatalf("WriteInitialLoRA bytes moved: SHA-256 %s, want %s (%d bytes)", digest, initialGoldenSHA256, len(body))
	}
}
