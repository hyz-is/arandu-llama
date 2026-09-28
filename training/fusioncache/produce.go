package fusioncache

import (
	"context"
	"errors"
	"fmt"
	"sort"
)

// PrefixRequest asks for the distribution after exactly Tokens, with no future
// gold tokens and no sampled continuation. Features refer to its final input row.
type PrefixRequest struct {
	Tokens       []int64
	PrefixSHA256 string
	TargetIndex  int
	Features     []FeatureSpec
}

// TeacherProbability is an unrenormalized probability in the teacher vocabulary.
type TeacherProbability struct {
	TokenID     int64
	Probability float64
}

// Signal echoes the executed prefix identity and contains no generated tokens.
type Signal struct {
	PrefixSHA256  string
	TargetIndex   int
	RetainedMass  float64
	Probabilities []TeacherProbability
	Features      []Feature
}

// Producer supplies teacher-forced signals and owns one resident teacher.
// Implementations must clear or prove equivalent prefix state for every request.
type Producer interface {
	Identity() ModelIdentity
	TeacherForce(context.Context, PrefixRequest) (Signal, error)
	Close() error
}

// Factory opens a typed backend for a pinned identity, never a shell command.
type Factory interface {
	Open(context.Context, ModelIdentity) (Producer, error)
}

// Receipt identifies one durable artifact. SHA256 covers its exact JSON bytes.
type Receipt struct {
	Location string
	SHA256   string
	Bytes    int64
}

// Sink persists a validated artifact before the next teacher is opened.
type Sink interface {
	Store(context.Context, Cache, Expectation, Limits) (Receipt, error)
}

// ProduceSequential opens at most one teacher, closes it even after failure,
// and requires successful closure and persistence before opening its successor.
// Returned receipts cover only fully completed teachers; no partial cache is saved.
func ProduceSequential(ctx context.Context, jobs []Expectation, factory Factory, sink Sink, limits Limits) ([]Receipt, error) {
	if ctx == nil || factory == nil || sink == nil || len(jobs) == 0 {
		return nil, fmt.Errorf("%w: context, jobs, factory and sink required", ErrContract)
	}
	// Freeze and validate all jobs before invoking any backend callbacks.
	frozen := make([]Expectation, len(jobs))
	for index, job := range jobs {
		if err := ValidateExpectation(job, limits); err != nil {
			return nil, err
		}
		data, err := encodeBounded(job, limits.MaxBytes)
		if err != nil {
			return nil, err
		}
		if err := decodeStrict(data, &frozen[index]); err != nil {
			return nil, err
		}
	}
	receipts := make([]Receipt, 0, len(jobs))
	for _, expected := range frozen {
		if err := ctx.Err(); err != nil {
			return receipts, err
		}
		producer, err := factory.Open(ctx, expected.Teacher)
		if err != nil {
			if producer != nil {
				err = errors.Join(err, producer.Close())
			}
			return receipts, err
		}
		if producer == nil {
			return receipts, fmt.Errorf("%w: factory returned nil producer", ErrContract)
		}
		cache, produceErr := produce(ctx, producer, expected, limits)
		closeErr := producer.Close()
		if err := errors.Join(produceErr, closeErr); err != nil {
			return receipts, err
		}
		if err := ctx.Err(); err != nil {
			return receipts, err
		}
		receipt, err := sink.Store(ctx, cache, expected, limits)
		if err != nil {
			return receipts, err
		}
		if !validSHA(receipt.SHA256) || receipt.Bytes <= 0 || receipt.Bytes > limits.MaxBytes || receipt.Location == "" {
			return receipts, fmt.Errorf("%w: invalid persistence receipt", ErrContract)
		}
		receipts = append(receipts, receipt)
	}
	return receipts, nil
}

func produce(ctx context.Context, p Producer, e Expectation, l Limits) (Cache, error) {
	if p.Identity() != e.Teacher {
		return Cache{}, fmt.Errorf("%w: resident teacher identity differs", ErrContract)
	}
	table, err := mappingTable(e, l)
	if err != nil {
		return Cache{}, err
	}
	c := Cache{Schema: 1, Mode: "teacher_forced", Teacher: e.Teacher, Student: e.Student, Mapping: e.Mapping, MappingSHA256: e.MappingSHA256, Features: e.Features}
	for _, x := range e.Examples {
		record := Record{Example: x}
		for position := x.PromptTokens; position < len(x.TeacherTokens); position++ {
			if err := ctx.Err(); err != nil {
				return Cache{}, err
			}
			prefix := PrefixDigest(x.TeacherTokens[:position])
			request := PrefixRequest{Tokens: append([]int64(nil), x.TeacherTokens[:position]...), PrefixSHA256: prefix, TargetIndex: position, Features: append([]FeatureSpec(nil), e.Features...)}
			signal, err := p.TeacherForce(ctx, request)
			if err != nil {
				return Cache{}, err
			}
			if p.Identity() != e.Teacher || signal.PrefixSHA256 != prefix || signal.TargetIndex != position || len(signal.Probabilities) == 0 || len(signal.Probabilities) > l.MaxTopK || len(signal.Features) != len(e.Features) {
				return Cache{}, fmt.Errorf("%w: teacher response identity or bounds differ", ErrContract)
			}
			row := Position{TargetIndex: position, TeacherPrefixSHA256: prefix, StudentPrefixSHA256: PrefixDigest(x.StudentTokens[:position]), RetainedMass: signal.RetainedMass}
			for _, probability := range signal.Probabilities {
				student, ok := mappedToken(table, probability.TokenID)
				if !ok {
					return Cache{}, fmt.Errorf("%w: unmapped teacher distribution token", ErrContract)
				}
				row.Probabilities = append(row.Probabilities, Probability{TeacherTokenID: probability.TokenID, StudentTokenID: student, Probability: probability.Probability})
			}
			sort.Slice(row.Probabilities, func(i, j int) bool { return row.Probabilities[i].TeacherTokenID < row.Probabilities[j].TeacherTokenID })
			for i, feature := range signal.Features {
				if feature.Spec != e.Features[i] || len(feature.Values) != feature.Spec.Dimension {
					return Cache{}, fmt.Errorf("%w: returned feature shape differs", ErrContract)
				}
				row.Features = append(row.Features, Feature{Spec: feature.Spec, Values: append([]float64(nil), feature.Values...)})
			}
			record.Positions = append(record.Positions, row)
		}
		c.Records = append(c.Records, record)
	}
	if err := ValidateAgainstStudent(c, e, l); err != nil {
		return Cache{}, err
	}
	return c, nil
}
