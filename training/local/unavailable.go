//go:build !libtorch || !cgo

package local

import "context"

// Run refuses native calculation without the required backend.
func Run(context.Context, Config) (Progress, error) { return Progress{}, ErrUnavailable }

// MeasureLosses refuses native calculation without the required backend.
func MeasureLosses(context.Context, LossConfig, []LossRequest) ([]LossResult, error) {
	return nil, ErrUnavailable
}

func runStageDelivery(context.Context, Config, *stepHooks) error { return ErrUnavailable }
