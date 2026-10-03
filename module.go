// Package llama is an Arandu module: one entity, one policy that decides
// about it, one service that owns its Model-first data path, and the routes that
// reach them.
//
// The files are laid out by role rather than by layer, so the whole package
// reads top to bottom:
//
//	module.go      -> registration, routes, handlers and migrations
//	config.go      -> what the application passes in
//	model.go       -> the entity, and what it may answer with
//	LlamaQuery.go  -> the generated query of the entity, never edited
//	policy.go      -> who may do what
//	service.go     -> the rules and Model access, after authorization
//	views.go       -> the files the application takes ownership of
//
// An application registers it explicitly. There is no service provider, no
// container and no discovery: the wiring is three lines somebody wrote, and
// reading them is how they learn what the application is made of.
package llama

import (
	"context"
	"errors"
	"fmt"
	stdhttp "net/http"
	"strconv"
	"strings"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/framework/validation"
	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
	"github.com/arandu-io/hesape/view"
)

// Module is what the application registers.
//
// It implements foundation.Module, which is Name and Routes and nothing else --
// that pair is the whole public contract between a package and the framework.
//
// It also implements foundation.Migratable, because it owns a table, and
// foundation.Publishable, because it hands view sources to the project. The
// other optional interfaces are declared beside Module in the framework and are
// opted into the same way, by implementing them: Bootable to prepare state at
// boot, Background to run a loop of its own, Schedulable to declare work for
// the scheduler, Health to report on the storage it depends on, Closable to
// give resources back at shutdown.
type Module struct {
	cfg      Config
	svc      *LlamaService
	sessions *security.SessionStore
}

// Compile-time proof that the module honors the contracts it claims.
var (
	_ foundation.Module      = (*Module)(nil)
	_ foundation.Migratable  = (*Module)(nil)
	_ foundation.Bootable    = (*Module)(nil)
	_ foundation.Publishable = (*Module)(nil)
)

// New returns the module, or the reason it cannot be built.
//
// The collaborators are parameters and not fields somebody fills in afterwards:
// a module that could be registered half-wired is a module whose first request
// is the thing that reports the missing half.
//
// It returns an error rather than panicking or carrying on, because everything
// it refuses is a wiring mistake, and a wiring mistake found at boot costs one
// restart. The same mistake found later is a request that reached a nil handle.
func New(cfg Config, db *data.DB, sessions *security.SessionStore) (*Module, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if db == nil {
		return nil, errors.New("llama: New needs a database handle: this package owns a table, and there is no in-memory mode that would let it start without one")
	}
	if sessions == nil {
		return nil, errors.New("llama: New needs a session store: it is where the subject comes from, and a request with no subject cannot be authorized")
	}
	cfg = cfg.withDefaults()
	return &Module{
		cfg:      cfg,
		svc:      NewLlamaService(db),
		sessions: sessions,
	}, nil
}

// Name is the module identifier: a lowercase slug, stable, no spaces.
//
// It is what `aru route:list` groups by and what the route names are prefixed with,
// so changing it changes addresses that other code has already written down.
func (m *Module) Name() string { return "llama" }

// Routes registers the module's routes under the configured prefix.
//
// They are named, so a URL is built from a name rather than written out a
// second time somewhere else -- two spellings of one address disagree, and the
// failure when they do is a link to a 404.
func (m *Module) Routes(r *fhttp.Router) {
	r.Action(stdhttp.MethodGet, m.cfg.Prefix, m.index).Name("llama.index")
	r.Action(stdhttp.MethodGet, m.cfg.Prefix+"/{id}", m.show).Name("llama.show")
	r.Action(stdhttp.MethodPost, m.cfg.Prefix, m.store).Name("llama.store")
}

// PublishCommand is what an application runs to take ownership of the views
// this package offers.
//
// It is spelled out as a constant so that whatever says it -- the refusal
// below, or a message an application writes for its own operators -- says one
// thing. A person who is told two different commands for one job tries both.
//
// The command reads the modules the application registered and writes what each
// one declares, which is why it is the application's command and not this
// package's: nothing outside the application knows which modules it holds.
const PublishCommand = "aru vendor:publish --apply"

// Boot refuses to serve when a view this package renders is not in the binary.
//
// A compiled view registers itself from init(), so by the time anything boots
// the question has one answer already: either the application published the
// files, compiled them and imported the package they became, or it did not.
// Asking here turns "did anybody run the install command" into one refusal at
// start-up that names the views and the command, instead of a 500 on the first
// request that reached one of them -- which is where it used to be answered,
// once per page, to whoever happened to open it.
//
// It also holds the destination. Every file the archive offers has to land
// under the module directory named after this module: an archive that reached
// resources/views/home.kyse.go would land on a page the application wrote, and
// what publishes the files writes what the archive says.
func (m *Module) Boot(context.Context) error {
	prefix := viewRoot + "/" + moduleDir + "/" + m.Name() + "/"
	var stray []string
	for _, path := range PublishedPaths() {
		if !strings.HasPrefix(path, prefix) {
			stray = append(stray, path)
		}
	}
	if len(stray) > 0 {
		return fmt.Errorf("llama: %s would be published outside %s, where it lands on a file the application wrote",
			strings.Join(stray, ", "), prefix)
	}

	registered := make(map[string]bool)
	for _, name := range view.Registered() {
		registered[name] = true
	}
	var missing []string
	for _, name := range ViewNames() {
		if !registered[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		// The compiled packages are named because the import is the half of the
		// install nothing else can do: the command writes the sources, the view
		// compiler turns them into Go, and a package the application does not
		// import is not linked at all.
		imports := make([]string, 0, len(ViewPackages()))
		for _, pkg := range ViewPackages() {
			imports = append(imports, "<module path>/"+pkg)
		}
		return fmt.Errorf("llama: no view is registered as %s. Run `%s`, then `aru view:build`, then import %s in bootstrap/app.go",
			strings.Join(missing, ", "), PublishCommand, strings.Join(imports, ", "))
	}
	return nil
}

// Handlers are thin on purpose: read the input, ask the service, answer. No
// rule, database handle or Model construction lives here. A handler that
// reached data directly would skip the service's policy boundary, and the
// layout makes that visible rather than relying on review.

// index answers a page of records.
func (m *Module) index(ctx *fhttp.Context) error {
	query := data.Query{
		Sort:   ctx.Query("sort"),
		Cursor: ctx.Query("cursor"),
		Limit:  m.cfg.PageSize,
	}

	records, err := m.svc.List(ctx.Ctx(), m.subject(ctx.Request), query)
	if err != nil {
		return m.answer(ctx, err)
	}

	// A full page is the only one that can have a successor. A short page is
	// the last one, and offering a cursor for it would be offering a next page
	// that comes back empty.
	cursor := ""
	if len(records) == m.cfg.PageSize {
		cursor = records[len(records)-1].ID
	}
	return ctx.JSON(stdhttp.StatusOK, collectionFromPointers(records, cursor))
}

// show answers one record.
func (m *Module) show(ctx *fhttp.Context) error {
	record, err := m.svc.Find(ctx.Ctx(), m.subject(ctx.Request), ctx.Param("id"))
	if err != nil {
		return m.answer(ctx, err)
	}
	return ctx.JSON(stdhttp.StatusOK, resourceFromPointer(record))
}

// store creates one record.
func (m *Module) store(ctx *fhttp.Context) error {
	// Every field is read from the request rather than defaulted. A reading with
	// a defaulted subject, policy or quantisation is a measurement that cannot be
	// compared with any other, which is the only thing a reading is for.
	loss, _ := strconv.ParseFloat(ctx.Input("loss"), 64)
	tokens, _ := strconv.Atoi(ctx.Input("tokens"))
	milliseconds, _ := strconv.ParseInt(ctx.Input("milliseconds"), 10, 64)
	in := CreateRequest{
		Name:         ctx.Input("name"),
		Subject:      ctx.Input("subject"),
		Policy:       ctx.Input("policy"),
		Quantisation: ctx.Input("quantisation"),
		Loss:         loss,
		Tokens:       tokens,
		Milliseconds: milliseconds,
	}

	record, err := m.svc.Create(ctx.Ctx(), m.subject(ctx.Request), in)
	if err != nil {
		return m.answer(ctx, err)
	}
	return ctx.JSON(stdhttp.StatusCreated, resourceFromPointer(record))
}

// subject reads who is acting from the session, and from nowhere else.
//
// A request with no readable session is a declared guest and not an empty
// subject. The difference matters: an empty subject is refused before the
// policy is consulted, because it is almost always a session that failed to
// load, and a policy asked about nobody answers about nobody. A guest reaches
// the policy and is refused there, by a rule somebody wrote -- or allowed,
// where the package means to serve a reader who never signed in.
//
// The tenant of that guest is the application's, from configuration. It is the
// one place a tenant does not come from a Grant, and it is because there is no
// Grant yet: everywhere downstream, data.Tenant is what the statements take.
func (m *Module) subject(r *stdhttp.Request) security.Subject {
	sub, err := m.sessions.Load(r.Context(), r)
	if err != nil || sub.ID == "" {
		return security.Guest(m.cfg.Tenant)
	}
	return sub
}

// answer turns what the service refused into something the client can act on.
//
// Three refusals have an answer, and everything else does not. An error this
// package did not expect is returned rather than swallowed: the framework turns
// it into the error page in development and a 500 in production, which is the
// honest outcome. Answering 200 with an empty body is the failure nobody
// debugs.
//
// A refusal is answered with a status and no detail. Telling the client why a
// policy said no is telling them what exists and what does not, one request at
// a time; the reason is in the log, where the person operating the system reads
// it and the person probing it does not.
func (m *Module) answer(ctx *fhttp.Context, err error) error {
	switch {
	case errors.Is(err, security.ErrForbidden):
		fhttp.Refuse(ctx.Response, ctx.Request, stdhttp.StatusForbidden, "forbidden")
		return nil
	case errors.Is(err, ErrNotFound):
		fhttp.Refuse(ctx.Response, ctx.Request, stdhttp.StatusNotFound, "not found")
		return nil
	}

	// A rejected input is the answer rather than a failure, and the fields that
	// were rejected are the client's own, so naming them gives nothing away.
	var rejected validation.Errors
	if errors.As(err, &rejected) {
		fhttp.Refuse(ctx.Response, ctx.Request, stdhttp.StatusUnprocessableEntity, rejected.Error())
		return nil
	}
	return err
}

// Migrations declares the schema this module owns.
//
// They are returned in the order their names sort in, which is the order they
// apply in: the name carries the order, and nothing else decides it.
func (m *Module) Migrations() []foundation.Migration {
	return []foundation.Migration{createLlamas{}}
}

// The migration is reversible, and the assertion is here rather than discovered
// at rollback: the migrator tests for Down with a type assertion, so a Down
// with the wrong signature would leave a rollback that silently does nothing.
var _ migrations.ReversibleMigration = createLlamas{}

// createLlamas is the table this module owns, and the index its listing
// reads by.
type createLlamas struct{ migrations.BaseMigration }

// GetName is the migration's identity, and it carries the order. It is fixed
// once the package is published: changing what an applied name means leaves the
// change missing everywhere it already ran, and nothing says so.
func (createLlamas) GetName() string { return "20260823_0001_create_llamas" }

// Up creates the table and the index the keyset pagination scans.
//
// The Blueprint spells each column for the engine the migration is running on,
// which is what lets one application develop on a file and deploy on Postgres
// without a second schema. That used to be written out as SQL here, with a
// comment explaining that an identifier column is VARCHAR rather than TEXT
// because it takes part in a key and MySQL refuses TEXT in one without a prefix
// length -- which is the grammar's job, done by hand in every migration that
// remembered to.
//
// The timestamp has no database default: the value comes from Go.
func (createLlamas) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, "llamas", func(table *schema.Blueprint) {
		table.String("id").Primary()
		table.String("tenant_id")
		table.String("name")
		// What was scored, under which adapter, at which bit-width. Two readings
		// are comparable only when they agree on the subject and differ on one of
		// the other two, so all three are columns rather than something
		// reconstructed from a name.
		table.String("subject")
		table.String("policy")
		table.String("quantisation")
		// The sum and the count, never the mean. A caller combining readings has
		// to weight each by its length, and a stored mean makes that impossible to
		// do correctly after the fact.
		table.Float("loss")
		table.Integer("tokens")
		table.BigInteger("milliseconds")
		table.Timestamp("created_at")

		// The index matches the ORDER BY of the listing, tenant first. Without
		// it every page is a scan of every customer's rows.
		table.Index([]string{"tenant_id", "created_at", "id"}, "llamas_tenant_created_idx")

		// The question asked of this table is almost always "how did the loss on
		// this subject move as the policy or the bit-width changed". Without this
		// index that is a scan, and it is the query a training run makes after
		// every step.
		table.Index([]string{"tenant_id", "subject", "created_at"}, "llamas_tenant_subject_idx")
	})
}

// Down drops the table, which takes its index with it.
func (createLlamas) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, "llamas")
}
