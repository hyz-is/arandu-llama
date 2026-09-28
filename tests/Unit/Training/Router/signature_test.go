package router_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/router"
)

// source parses the router package's non-test files.
func source(t *testing.T) (*token.FileSet, []*ast.File) {
	t.Helper()
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate the test file")
	}
	dir := filepath.Join(filepath.Dir(here), "..", "..", "..", "..", "training", "router")
	matches, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil || len(matches) == 0 {
		t.Fatalf("no router source under %s: %v", dir, err)
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, path := range matches {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, file)
	}
	return fset, files
}

// TestNoExportedFunctionCanReceiveATeacherOutput holds the property that makes
// the decision prior to generation: every exported entry point takes only the
// configuration, the teacher table, the example's identity, or the bytes and
// digest of a registered configuration. None of those types can carry what a
// teacher produced, and TestTheRoutedExampleCarriesOnlyItsIdentity pins the
// one that describes the example.
func TestNoExportedFunctionCanReceiveATeacherOutput(t *testing.T) {
	allowed := []string{"Config", "[]Teacher", "Example", "string", "[]byte"}
	fset, files := source(t)
	exported := 0
	for _, file := range files {
		for _, declaration := range file.Decls {
			fn, ok := declaration.(*ast.FuncDecl)
			if !ok || !fn.Name.IsExported() {
				continue
			}
			exported++
			for _, field := range fn.Type.Params.List {
				if kind := types.ExprString(field.Type); !slices.Contains(allowed, kind) {
					t.Errorf("%s: %s accepts %s, which is not one of %v", fset.Position(fn.Pos()), fn.Name.Name, kind, allowed)
				}
			}
		}
	}
	if exported < 4 {
		t.Fatalf("found %d exported functions; the parser is not reading the package", exported)
	}
}

func TestTheRoutedExampleCarriesOnlyItsIdentity(t *testing.T) {
	kind := reflect.TypeFor[router.Example]()
	var fields []string
	for i := range kind.NumField() {
		fields = append(fields, kind.Field(i).Name)
	}
	if !slices.Equal(fields, []string{"ID", "Subcapability"}) {
		t.Fatalf("Example has fields %v; anything beyond its identity could carry a generation", fields)
	}
}

func TestTheRouterImportsOnlyTheStandardLibrary(t *testing.T) {
	_, files := source(t)
	for _, file := range files {
		for _, spec := range file.Imports {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				t.Fatal(err)
			}
			if first := strings.Split(path, "/")[0]; strings.Contains(first, ".") {
				t.Errorf("the router imports %s; its decision type stands alone", path)
			}
		}
	}
}
