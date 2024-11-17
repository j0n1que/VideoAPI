package main

import (
	"fmt"
	"net/http"

	"github.com/j0n1que/VideoAPI/API/internal/transport/rest"
)

func main() {
	handler := rest.NewHandler()
	srv := &http.Server{
		Addr:    ":8080",
		Handler: handler.InitRouter(),
	}
	if err := srv.ListenAndServe(); err != nil {
		fmt.Println(err)
	}
}
