package rest

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/gorilla/mux"
)

const (
	filePath = "../../../python/build"
)

type Handler struct {
}

func NewHandler() *Handler { return &Handler{} }

func (h *Handler) InitRouter() *mux.Router {

	r := mux.NewRouter()

	r.Use(loggingMiddleware)

	videos := r.PathPrefix("/").Subrouter()

	{
		videos.HandleFunc("/video", h.getVideo).Methods(http.MethodPost)
	}

	return r
}

func (h *Handler) getVideo(w http.ResponseWriter, r *http.Request) {
	r.ParseMultipartForm(32 << 20)
	file, handler, err := r.FormFile("video")
	if err != nil {
		http.Error(w, "Could not retrieve video from form", http.StatusBadRequest)
		return
	}
	defer file.Close()
	if handler.Header.Get("Content-Type") != "video/mp4" {
		http.Error(w, "Invalid file type, only MP4 is allowed", http.StatusUnsupportedMediaType)
		return
	}
	w.Header().Set("Content-Disposition", "inline; filename=video.mp4")
	var buffer bytes.Buffer
	_, err = io.Copy(&buffer, file)
	if err != nil {
		fmt.Println(err)
		return
	}
	err = runPython(buffer)
	if err != nil {
		http.Error(w, "Error analyzing video", http.StatusInternalServerError)
	}
	videoPath := filepath.Join(".", filePath+"/video_with_audio.mp4")
	videoFile, err := os.Open(videoPath)
	if err != nil {
		http.Error(w, "Error reading video", http.StatusInternalServerError)
		return
	}
	defer os.RemoveAll(filePath)
	defer videoFile.Close()
	videoData, err := io.ReadAll(videoFile)
	if err != nil {
		http.Error(w, "Error reading video content", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "video/mp4")
	if _, err := w.Write(videoData); err != nil {
		http.Error(w, "Error writing video to response", http.StatusInternalServerError)
		return
	}
}

func runPython(buffer bytes.Buffer) error {
	os.MkdirAll(filePath, os.ModePerm)
	cmd := exec.Command("python", "../../../python/infer_test.py")
	cmd.Stdin = bytes.NewReader(buffer.Bytes())
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	return err
}
