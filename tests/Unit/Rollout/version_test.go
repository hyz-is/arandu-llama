package rollout_test

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	llama "github.com/tayi-ai/arandu-llama"
)

// packageRoot is the module root, three directories above this file.
func packageRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("the location of this test file is unknown")
	}
	return filepath.Join(filepath.Dir(file), "..", "..", "..")
}

// RolloutVersion carries the llama.cpp commit because moving the pin moves the
// logits, and with them every log mu, while nothing else in a rollout changes.
func TestRolloutVersionNamesThePinnedEngine(t *testing.T) {
	t.Parallel()

	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH: the gitlink cannot be read")
	}
	command := exec.Command("git", "ls-tree", "HEAD", "llama.cpp")
	command.Dir = packageRoot(t)
	output, err := command.Output()
	if err != nil {
		t.Skipf("not a git checkout: %v", err)
	}
	fields := strings.Fields(string(output))
	if len(fields) < 3 || fields[1] != "commit" || len(fields[2]) < 8 {
		t.Skipf("llama.cpp is not a gitlink at HEAD: %q", strings.TrimSpace(string(output)))
	}
	want := "llama.cpp." + fields[2][:8]
	if !strings.HasSuffix(llama.RolloutVersion, want) {
		t.Fatalf("RolloutVersion %q does not end with %q: the submodule moved and the constant did not", llama.RolloutVersion, want)
	}
}
