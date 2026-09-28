package bridge

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
)

// Relation is how a student token sits against the realized teacher
// segmentation, the dispatch of the paper's Algorithm 1.
type Relation uint8

const (
	// RelationOneToOne starts and ends on the boundaries of one teacher token.
	RelationOneToOne Relation = iota + 1
	// RelationHead starts on a teacher boundary and ends inside that token
	// (the head of a 1:N split).
	RelationHead
	// RelationInterior starts and ends inside one teacher token whose end is a
	// sync point (the interior of a 1:N split).
	RelationInterior
	// RelationSpanning starts on a teacher boundary and crosses at least one
	// more (N:1).
	RelationSpanning
	// RelationMidSpanning starts inside a teacher token and crosses its end
	// (M:N).
	RelationMidSpanning
	// RelationExcluded starts inside a teacher token that ends inside another
	// student token: the row neither begins at nor reaches a sync point and
	// admits no target.
	RelationExcluded
)

// String names the relation.
func (r Relation) String() string {
	switch r {
	case RelationOneToOne:
		return "1:1"
	case RelationHead:
		return "1:N head"
	case RelationInterior:
		return "1:N interior"
	case RelationSpanning:
		return "N:1"
	case RelationMidSpanning:
		return "M:N"
	case RelationExcluded:
		return "excluded"
	}
	return fmt.Sprintf("relation(%d)", uint8(r))
}

// Alignment is one byte string as the student and the teacher realized it.
// Stop and control tokens carry no bytes and are stripped by the caller
// before aligning, as the paper does.
type Alignment struct {
	student, teacher []int64
	stream           string
	studentOffsets   []int
	teacherOffsets   []int
	sync             []bool
}

// Align checks that both token sequences decode to the same bytes and marks
// the sync points, the byte offsets where both segmentations have a boundary.
func (b *Bridge) Align(student, teacher []int64) (*Alignment, error) {
	studentStream, studentOffsets, err := concatenate(b.student, student, "student")
	if err != nil {
		return nil, err
	}
	teacherStream, teacherOffsets, err := concatenate(b.teacher, teacher, "teacher")
	if err != nil {
		return nil, err
	}
	for position, id := range teacher {
		if b.stops[id] {
			return nil, fmt.Errorf("%w: teacher stop token %d at position %d; strip stop tokens first", ErrQuery, id, position)
		}
	}
	if studentStream != teacherStream {
		return nil, fmt.Errorf("%w: student and teacher sequences decode to different bytes", ErrQuery)
	}
	a := &Alignment{
		student:        slices.Clone(student),
		teacher:        slices.Clone(teacher),
		stream:         studentStream,
		studentOffsets: studentOffsets,
		teacherOffsets: teacherOffsets,
		sync:           make([]bool, len(studentStream)+1),
	}
	teacherBoundary := make([]bool, len(teacherStream)+1)
	for _, offset := range teacherOffsets {
		teacherBoundary[offset] = true
	}
	for _, offset := range studentOffsets {
		a.sync[offset] = teacherBoundary[offset]
	}
	return a, nil
}

func concatenate(v *Vocabulary, ids []int64, side string) (string, []int, error) {
	if len(ids) == 0 {
		return "", nil, fmt.Errorf("%w: empty %s sequence", ErrQuery, side)
	}
	var stream strings.Builder
	offsets := make([]int, 0, len(ids)+1)
	offsets = append(offsets, 0)
	for position, id := range ids {
		s, ok := v.representation(id)
		if !ok {
			return "", nil, fmt.Errorf("%w: %s token %d at position %d has no bytes; strip stop and control tokens first", ErrQuery, side, id, position)
		}
		stream.WriteString(s)
		offsets = append(offsets, stream.Len())
	}
	return stream.String(), offsets, nil
}

// Students returns the number of student positions.
func (a *Alignment) Students() int { return len(a.student) }

// Teachers returns the number of teacher positions.
func (a *Alignment) Teachers() int { return len(a.teacher) }

// Sync reports whether byte offset is a sync point.
func (a *Alignment) Sync(offset int) bool {
	return offset >= 0 && offset < len(a.sync) && a.sync[offset]
}

// Row is the target of one student position.
type Row struct {
	// Position is the student position; Offset its first byte.
	Position int
	Offset   int
	// Teacher is the teacher position whose token holds Offset, and Prefix
	// how many of its bytes precede Offset.
	Teacher int
	Prefix  int
	// Relation is the row's case in the dispatch.
	Relation Relation
	// Masked is set when the student token is only spaces and tabs; BPM
	// withholds those rows unless the caller keeps them.
	Masked bool
	// Target is nil when the row admits no target: excluded, masked, or its
	// realized prefix has no teacher support.
	Target *Target
}

// RowOptions tunes Rows.
type RowOptions struct {
	// KeepWhitespace computes targets for whitespace-only student tokens,
	// which the paper masks by default.
	KeepWhitespace bool
}

// Rows computes the target of every student position. teacher[j] is the
// teacher's distribution for its token j, predicted from the tokens before
// it. A row at a teacher boundary takes Eq. 4; a row inside a teacher token
// takes Eq. 5 capped at the token's end, which is then the chunk boundary; a
// row whose token crosses a teacher boundary adds the chain correction of
// Eq. 6 and is marked as a lower bound. The cells of a spanning correction
// are keyed by the token Vocabulary.Find would return for their bytes.
func (b *Bridge) Rows(a *Alignment, teacher []Distribution, options RowOptions) ([]Row, error) {
	if a == nil || len(teacher) != len(a.teacher) {
		return nil, fmt.Errorf("%w: one teacher distribution per teacher position required", ErrQuery)
	}
	dists := make([]validated, len(teacher))
	for j, d := range teacher {
		entries, uncovered, err := b.validate(d)
		if err != nil {
			return nil, fmt.Errorf("teacher position %d: %w", j, err)
		}
		dists[j] = validated{entries, uncovered}
	}
	rows := make([]Row, len(a.student))
	for i, id := range a.student {
		start, end := a.studentOffsets[i], a.studentOffsets[i+1]
		k := sort.SearchInts(a.teacherOffsets, start+1) - 1
		teacherStart, teacherEnd := a.teacherOffsets[k], a.teacherOffsets[k+1]
		row := Row{Position: i, Offset: start, Teacher: k, Prefix: start - teacherStart}
		switch {
		case start == teacherStart && end == teacherEnd:
			row.Relation = RelationOneToOne
		case start == teacherStart && end < teacherEnd:
			row.Relation = RelationHead
		case start == teacherStart:
			row.Relation = RelationSpanning
		case end > teacherEnd:
			row.Relation = RelationMidSpanning
		case a.sync[teacherEnd]:
			row.Relation = RelationInterior
		default:
			row.Relation = RelationExcluded
		}
		row.Masked = !options.KeepWhitespace && b.student.WhitespaceOnly(id)
		if row.Relation == RelationExcluded || row.Masked {
			rows[i] = row
			continue
		}
		prefix := a.stream[teacherStart:start]
		limit := 0
		if prefix != "" {
			limit = teacherEnd - start
		}
		target, err := b.marginalize(prefix, limit, dists[k].entries, dists[k].uncovered)
		if errors.Is(err, ErrNoSupport) {
			rows[i] = row
			continue
		}
		if err != nil {
			return nil, err
		}
		if row.Relation == RelationSpanning || row.Relation == RelationMidSpanning {
			b.correct(&target, a, dists, k, start)
		}
		row.Target = &target
		rows[i] = row
	}
	return rows, nil
}

// correct applies the spanning correction to the base target of a row
// starting at byte start inside, or at the start of, teacher token k. Every
// student token that prefixes the realized bytes from start and is longer
// than the rest of token k is a candidate; each receives the realized-path
// chain value (Eq. 6), divided by the base target's support. The value of the
// shortest is taken from the head cell -- where the realized teacher token's
// own mass was routed -- and nested candidates keep telescoping differences,
// so the total mass does not change.
func (b *Bridge) correct(t *Target, a *Alignment, dists []validated, k, start int) {
	teacherStart, teacherEnd := a.teacherOffsets[k], a.teacherOffsets[k+1]
	type candidate struct {
		student int64
		value   float64
	}
	var candidates []candidate
	for n := teacherEnd - start + 1; n <= min(b.longest, len(a.stream)-start); n++ {
		u, ok := b.index[a.stream[start:start+n]]
		if !ok {
			continue
		}
		// The chain walks the realized teacher tokens from the start of token
		// k: exact probabilities for the covered tokens, prefix mass for the
		// token the candidate ends in.
		end := start + n
		value := 1.0
		for j, at := k, teacherStart; ; j++ {
			next := a.teacherOffsets[j+1]
			if end <= next {
				value *= b.prefixMass(dists[j].entries, a.stream[at:end])
				break
			}
			value *= probability(dists[j].entries, a.teacher[j])
			at = next
		}
		candidates = append(candidates, candidate{u, value / t.Support})
	}
	if len(candidates) == 0 {
		return
	}
	head := b.prefix(a.stream[start:teacherEnd], 0)
	moved := candidates[0].value
	t.Chain = moved
	t.Bound = BoundLowerBound
	cells := make([]Probability, 0, len(t.Cells)+len(candidates))
	cells = append(cells, t.Cells...)
	if head == NoToken {
		t.Residual = max(0, t.Residual-moved)
	} else {
		cells = addCell(cells, head, -moved)
	}
	for i, c := range candidates {
		delta := c.value
		if i+1 < len(candidates) {
			delta -= candidates[i+1].value
		}
		cells = addCell(cells, c.student, delta)
	}
	for i := range cells {
		// A difference of nested chain values is never negative; rounding
		// can leave one a few ulps below zero.
		cells[i].Probability = max(0, cells[i].Probability)
	}
	t.Cells = cells
}

// addCell adds delta to the cell of student token id, inserting it in id
// order when absent.
func addCell(cells []Probability, id int64, delta float64) []Probability {
	i, found := slices.BinarySearchFunc(cells, id, func(c Probability, id int64) int { return cmp.Compare(c.Token, id) })
	if found {
		cells[i].Probability += delta
		return cells
	}
	return slices.Insert(cells, i, Probability{id, delta})
}

// probability returns the retained probability of teacher token id, zero
// when the top-k does not hold it.
func probability(entries []entry, id int64) float64 {
	if i, ok := slices.BinarySearchFunc(entries, id, func(e entry, id int64) int { return cmp.Compare(e.token, id) }); ok {
		return entries[i].p
	}
	return 0
}

// prefixMass returns the retained mass of teacher tokens whose bytes begin
// with s, summed in teacher-id order.
func (b *Bridge) prefixMass(entries []entry, s string) float64 {
	sum := 0.0
	for _, e := range entries {
		if b.stops[e.token] {
			continue
		}
		if bytes, ok := b.teacher.representation(e.token); ok && strings.HasPrefix(bytes, s) {
			sum += e.p
		}
	}
	return sum
}
