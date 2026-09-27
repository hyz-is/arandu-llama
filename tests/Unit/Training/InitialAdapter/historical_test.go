//go:build libtorch && cgo

package initialadapter_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
)

// historicalSpec is the q/v initializer every earlier local run was pinned
// to: rank 4 on the eight full-attention layers of a 32-layer hybrid decoder.
func historicalSpec() decoder.InitialAdapterSpec {
	s := decoder.InitialAdapterSpec{Seed: 83, PreludeBlocks: 24, PreludeWidth: 32, PreludeLow: .01, PreludeHigh: 16,
		ExpectedSHA256: "75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c"}
	for layer := 3; layer < 32; layer += 4 {
		for _, projection := range []struct {
			name   string
			output int64
		}{{"q_proj", 8192}, {"v_proj", 1024}} {
			s.Projections = append(s.Projections, decoder.InitialProjection{Name: fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.%s", layer, projection.name), Input: 4096, Output: projection.output, Rank: 4})
		}
	}
	return s
}

// Any change to the schedule, the stream or the digest encoding fails here
// before it can reach a run that is still pinned to the historical bytes.
func TestHistoricalInitializerKeepsItsPin(t *testing.T) {
	spec := historicalSpec()
	a, err := decoder.InitializeAdapter(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if a.SHA256 != spec.ExpectedSHA256 || len(a.Parameters) != 32 {
		t.Fatal("historical initializer identity differs", a.SHA256, len(a.Parameters))
	}
}
