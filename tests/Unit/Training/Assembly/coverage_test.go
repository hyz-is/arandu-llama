package assembly_test

import (
	"crypto/sha256"
	"fmt"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
)

// layoutDigest hashes every admitted tensor in plan order: names, reference
// identity, geometry, precision, placement and trainability. Shards are left
// out: the fixture assigns them in map order.
func layoutDigest(p *decoder.AssemblyPlan) string {
	h := sha256.New()
	for _, item := range p.Tensors() {
		fmt.Fprintf(h, "%s|%s|%s|%v|%d|%v|%d|%t\n", item.ReferenceName, item.SourceName, item.SHA256, item.Shape, item.DType, item.Device, item.Bytes, item.Trainable)
	}
	summary := p.Summary()
	fmt.Fprintf(h, "%d|%d|%d|%v\n", summary.BaseTensors, summary.AdapterTensors, summary.AdapterElements, summary.PersistentBytes)
	return fmt.Sprintf("%x", h.Sum(nil))
}

// The q/v plan predates coverage declarations; any byte of its layout that
// moves fails here.
func TestPlanWithoutCoverageKeepsItsLayout(t *testing.T) {
	if digest := layoutDigest(plan(t, fixture())); digest != "9c8f4149e26aa9cdb4ddf9db0ae724a9ecb6c989122491b3b3fd6af09010ef44" {
		t.Fatalf("the q/v plan layout changed: %s", digest)
	}
}
