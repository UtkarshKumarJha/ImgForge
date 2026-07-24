import { useRef, useState } from "react";
import {
  Upload,
  Search,
  ShieldCheck,
  ScanSearch,
  Image as ImageIcon,
  AlertTriangle,
  CheckCircle2,
  X,
} from "lucide-react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const selectFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handleFileChange = (event) => {
    selectFile(event.target.files[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    selectFile(event.dataTransfer.files[0]);
  };

  const removeImage = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Upload an image before starting analysis.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis request failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(
        "Unable to analyze the image. Make sure the ImgForge API is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <ScanSearch size={23} />
          </div>

          <div>
            <h2>ImgForge</h2>
            <span>Image Forensics AI</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI Engine Ready
        </div>
      </nav>

      <main className="container">
        <section className="hero">
          <div className="hero-badge">
            <ShieldCheck size={15} />
            Deep Learning Image Forensics
          </div>

          <h1>
            Detect Image <span>Manipulation</span>
          </h1>

          <p>
            Upload an image and ImgForge analyzes visual and compression
            artifacts using EfficientNet-B0, Error Level Analysis and Grad-CAM.
          </p>
        </section>

        <section className="workspace">
          {result && (
  <section className="visual-analysis">

    <div className="visual-header">

      <div>
        <p className="eyebrow">
          EXPLAINABLE AI
        </p>

        <h2>
          Forensic Visualization
        </h2>
      </div>

      <span>
        ELA + Grad-CAM
      </span>

    </div>

    <div className="visual-grid">

      {/* ORIGINAL */}
      <div className="visual-card">

        <div className="visual-title">
          <div>
            <p>INPUT</p>
            <h3>Original Image</h3>
          </div>
        </div>

        <div className="visual-image-container">
          <img
            src={preview}
            alt="Original"
          />
        </div>

        <p className="visual-description">
          Original image submitted to the
          forensic pipeline.
        </p>

      </div>


      {/* ELA */}
      <div className="visual-card">

        <div className="visual-title">
          <div>
            <p>FORENSIC SIGNAL</p>
            <h3>Error Level Analysis</h3>
          </div>
        </div>

        <div className="visual-image-container">
          <img
            src={result.ela_image}
            alt="ELA analysis"
          />
        </div>

        <p className="visual-description">
          Highlights JPEG compression
          inconsistencies that may indicate
          edited regions.
        </p>

      </div>


      {/* GRAD CAM */}
      <div className="visual-card">

        <div className="visual-title">
          <div>
            <p>MODEL EXPLAINABILITY</p>
            <h3>Grad-CAM</h3>
          </div>
        </div>

        <div className="visual-image-container">
          <img
            src={result.gradcam_image}
            alt="Grad-CAM analysis"
          />
        </div>

        <p className="visual-description">
          Shows regions that contributed most
          strongly toward the forged prediction.
        </p>

      </div>

    </div>

  </section>
)}
          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">INPUT</p>
                <h2>Upload Image</h2>
              </div>

              <ImageIcon size={22} />
            </div>

            {!preview ? (
              <div
                className="drop-zone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
              >
                <div className="upload-icon">
                  <Upload size={28} />
                </div>

                <h3>Drop your image here</h3>
                <p>or click to browse from your computer</p>

                <button type="button" className="choose-button">
                  Choose Image
                </button>

                <span>JPEG, JPG, PNG supported</span>
              </div>
            ) : (
              <div className="preview-container">
                <button className="remove-button" onClick={removeImage}>
                  <X size={18} />
                </button>

                <img src={preview} alt="Selected" className="preview-image" />

                <div className="file-details">
                  <div>
                    <strong>{file.name}</strong>
                    <p>{(file.size / 1024).toFixed(1)} KB</p>
                  </div>

                  <CheckCircle2 size={21} />
                </div>
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg"
              hidden
              onChange={handleFileChange}
            />

            <button
              className="analyze-button"
              onClick={analyzeImage}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing forensic signals...
                </>
              ) : (
                <>
                  <Search size={19} />
                  Analyze Image
                </>
              )}
            </button>

            {error && (
              <div className="error-message">
                <AlertTriangle size={17} />
                {error}
              </div>
            )}
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">OUTPUT</p>
                <h2>Forensic Analysis</h2>
              </div>

              <Search size={22} />
            </div>

            {!result && !loading && (
              <div className="empty-result">
                <div className="scan-graphic">
                  <ScanSearch size={44} />
                </div>

                <h3>Awaiting analysis</h3>
                <p>
                  Upload an image and run the forensic engine to view the
                  results.
                </p>
              </div>
            )}

            {loading && (
              <div className="empty-result">
                <div className="scanner"></div>
                <h3>Analyzing image</h3>
                <p>Inspecting image structure and forensic signals...</p>
              </div>
            )}

            {result && (
  <div className="results forensic-results">

    {/* VERDICT */}
    <div
      className={`verdict-card ${
        result.verdict === "FORGED" ? "forged" : "authentic"
      }`}
    >
      <div className="verdict-icon">
        {result.verdict === "FORGED" ? (
          <AlertTriangle size={28} />
        ) : (
          <CheckCircle2 size={28} />
        )}
      </div>

      <div>
        <p>FORENSIC VERDICT</p>

        <h2>{result.verdict}</h2>

        <span>
          {result.confidence}% confidence
        </span>
      </div>
    </div>

    {/* PROBABILITY */}
    <div className="probability-section">

      <div className="probability-heading">
        <span>Classification Confidence</span>
        <strong>{result.confidence}%</strong>
      </div>

      <div className="confidence-track">
        <div
          className={`confidence-fill ${
            result.verdict === "FORGED"
              ? "forged-fill"
              : "authentic-fill"
          }`}
          style={{
            width: `${result.confidence}%`,
          }}
        />
      </div>

    </div>

    {/* AUTHENTIC / FORGED PROBABILITIES */}
    <div className="probability-grid">

      <div className="probability-card">
        <p>AUTHENTIC</p>
        <strong>
          {result.authentic_probability}%
        </strong>
      </div>

      <div className="probability-card">
        <p>FORGED</p>
        <strong>
          {result.forged_probability}%
        </strong>
      </div>

    </div>

    {/* TAMPER LOCALIZATION */}
    <div className="tamper-card">

      <p>TAMPER LOCALIZATION</p>

      {result.verdict === "FORGED" ? (
        <>
          <h3>{result.tamper_zone}</h3>

          <span>
            Suspicious activation detected in this region
          </span>
        </>
      ) : (
        <>
          <h3>No significant tampering detected</h3>

          <span>
            Image classified as authentic
          </span>
        </>
      )}

    </div>

    {/* MODEL */}
    <div className="model-info">

      <span>
        Model
        <strong>{result.model}</strong>
      </span>

      <span>
        Device
        <strong>
          {result.device?.toUpperCase()}
        </strong>
      </span>

    </div>

  </div>
)}
          </div>
        </section>

        <section className="technology">
          <span>EfficientNet-B0</span>
          <span>Error Level Analysis</span>
          <span>Grad-CAM</span>
          <span>PyTorch</span>
          <span>FastAPI</span>
        </section>
      </main>

      <footer>
        <p>ImgForge · Deep Learning Image Forensics</p>
      </footer>
    </div>
  );
}

export default App;