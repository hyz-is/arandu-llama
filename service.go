package llama

import (
	"context"
	"fmt"
	"math"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/framework/validation"
)

// Pagination bounds for List. A request that asks for everything gets the
// maximum, never everything: an unbounded query is how one page load takes a
// production database down.
const (
	defaultLimit = 50
	maxLimit     = 200
)

// sortableLlama is the ordering allowlist. A column name taken directly
// from a request would turn ordering into an injection surface.
var sortableLlama = map[string]string{
	"":           "created_at",
	"name":       "name",
	"created_at": "created_at",
}

// LlamaService holds the rules of this package.
//
// It receives its collaborators through the constructor. There is no container
// and no resolution by reflection: what this service is made of is written at
// the one place that builds it, and reading that place is how somebody learns
// what the package touches.
//
// Everything a handler is allowed to do goes through here. The service is the
// only owner of the database handle, so the request layer cannot reach a Model
// before the policy has answered.
type LlamaService struct {
	db     *data.DB
	policy LlamaPolicy
}

// NewLlamaService wires the service over the application's database handle.
func NewLlamaService(db *data.DB) *LlamaService {
	return &LlamaService{db: db}
}

// CreateRequest is the input contract.
//
// The fields are explicit and there is no mass assignment, so a request body
// cannot write a column nobody meant to expose. There is no TenantID here and
// there must never be one: the tenant comes from the Grant, which comes from
// the session.
type CreateRequest struct {
	// Name is what the record will be called.
	Name string
	// Subject is what was scored, in the caller's own vocabulary.
	Subject string
	// Policy is the digest of the adapter that was applied.
	Policy string
	// Quantisation is the representation the weights were in.
	Quantisation string
	// Loss is the summed negative log likelihood; Tokens is how many positions
	// it covers.
	Loss   float64
	Tokens int
	// Milliseconds is how long the engine took.
	Milliseconds int64
}

// Validate reports the errors per field.
func (r CreateRequest) Validate() validation.Errors {
	e := validation.Errors{}
	validation.Required(e, "name", r.Name)
	validation.MaxLen(e, "name", r.Name, 120)
	validation.Required(e, "subject", r.Subject)
	validation.MaxLen(e, "subject", r.Subject, 200)
	validation.Required(e, "policy", r.Policy)
	validation.MaxLen(e, "policy", r.Policy, 128)
	validation.Required(e, "quantisation", r.Quantisation)
	validation.MaxLen(e, "quantisation", r.Quantisation, 32)
	// Tokens has to be positive and Loss finite, and neither is a formality. A
	// reading of zero positions has a mean of zero, and zero reads downstream as
	// a perfect prediction rather than as an absent measurement. A non-finite
	// loss propagates through every average it enters and arrives as a decision
	// nobody can trace back.
	if r.Tokens <= 0 {
		e.Add("tokens", "a reading has to cover at least one position; a mean over none reads as a perfect prediction")
	}
	if math.IsNaN(r.Loss) || math.IsInf(r.Loss, 0) {
		e.Add("loss", "the loss is not finite, and would poison every average it enters")
	}
	if r.Milliseconds < 0 {
		e.Add("milliseconds", "a reading cannot have taken negative time")
	}
	return e
}

// Compile-time proof that the request honors the validation contract.
var _ validation.Validatable = CreateRequest{}

// Create is the whole path in one function: validate, authorize, then act with
// the Grant the authorization produced.
//
// The candidate is authorized before it is stored, and the candidate is what
// the policy sees -- so a rule about what may be created is a rule about the
// record being created, and not about the person alone.
func (s *LlamaService) Create(ctx context.Context, actor security.Subject, in CreateRequest) (*Llama, error) {
	if errs := in.Validate(); errs.Any() {
		return nil, errs
	}

	proposed := Llama{
		Name: in.Name, Subject: in.Subject, Policy: in.Policy,
		Quantisation: in.Quantisation, Loss: in.Loss, Tokens: in.Tokens,
		Milliseconds: in.Milliseconds,
	}

	g, err := security.Authorize(ctx, s.policy, actor, LlamaCreate, proposed)
	if err != nil {
		return nil, err
	}
	if proposed.ID, err = data.NewID(); err != nil {
		return nil, err
	}
	candidate, err := Llamas(s.db).New()
	if err != nil {
		return nil, err
	}
	candidate.ID = proposed.ID
	candidate.TenantID = data.Tenant(g)
	candidate.Name = proposed.Name
	if _, err := candidate.Save(ctx, g); err != nil {
		return nil, err
	}
	return candidate, nil
}

// Find returns one record, and asks the policy twice.
//
// The first call is on the empty candidate, because there is no way to read the
// record without a Grant and no way to hold a Grant without a decision. What it
// decides is whether this subject may view records of this kind at all.
//
// The second call is on the record that came back, and it is the one a rule
// about the record itself depends on: the first call saw an empty value, so
// anything the policy says about who owns the row, or about a row that is not
// published yet, never ran. Without it a policy can be written that looks
// correct, reads correctly, and is never consulted about the thing it protects.
//
// The read itself is already scoped by data.Tenant, so the second call is not
// what keeps customers apart. It is what keeps the policy honest.
func (s *LlamaService) Find(ctx context.Context, actor security.Subject, id string) (*Llama, error) {
	g, err := security.Authorize(ctx, s.policy, actor, LlamaView, Llama{})
	if err != nil {
		return nil, err
	}

	record, err := Llamas(s.db).WhereKey(id).First(ctx, g)
	if err != nil {
		return nil, err
	}
	if record == nil {
		return nil, ErrNotFound
	}

	if _, err := security.Authorize(ctx, s.policy, actor, LlamaView, *record); err != nil {
		return nil, err
	}
	return record, nil
}

// List returns a page of records.
//
// It authorizes once, on the empty candidate, and the tenant filter in the
// statement is what bounds the rows. A policy call per row would be one call
// per record on a page and would still not narrow the query -- a listing that
// has to read a customer's rows in order to decide it may not read them has
// already read them.
//
// A rule that hides individual records from a listing belongs in the statement,
// as a predicate, and the action here is what decides whether the listing may
// run at all.
func (s *LlamaService) List(ctx context.Context, actor security.Subject, q data.Query) ([]*Llama, error) {
	g, err := security.Authorize(ctx, s.policy, actor, LlamaList, Llama{})
	if err != nil {
		return nil, err
	}

	column, ok := sortableLlama[q.Sort]
	if !ok {
		return nil, fmt.Errorf("llama: sort field not allowed: %q", q.Sort)
	}

	limit := q.Limit
	switch {
	case limit <= 0:
		limit = defaultLimit
	case limit > maxLimit:
		limit = maxLimit
	}

	page := Llamas(s.db)
	if q.Cursor != "" {
		anchor, err := Llamas(s.db).WhereKey(q.Cursor).Value(ctx, g, column)
		if err != nil {
			return nil, err
		}
		if anchor == nil {
			return nil, nil
		}
		page = page.Where(func(after *LlamaQuery) {
			after.Where(column, ">", anchor).
				OrWhere(func(equal *LlamaQuery) {
					equal.Where(column, "=", anchor).Where("id", ">", q.Cursor)
				})
		})
	}

	return page.OrderBy(column).OrderBy("id").Limit(limit).Get(ctx, g)
}
