from flask import Flask, request, jsonify
from flask_cors import CORS
from security_agent import SecurityAgent
from models import Signal, SecurityContext, EventType
from datetime import datetime
import tempfile
import base64
import os
import wave
import io

# Import the REAL AI functions from your pipeline files
from video_analysis_pipeline import (
    vision_analyse,
    pick_vision_model,
    available_models,
    extract_frames,
    ollama_running,  # ← ADD THIS IMPORT
)
from audio_analysis_pipeline import analyse_audio as run_audio_analysis
from multimodal_pipeline import fuse_signals, officer_briefing

# CREATE THE FLASK APP (this was the issue!)
app = Flask(__name__)
CORS(app)


def process_audio_base64(audio_base64: str) -> dict:
    """
    Convert base64 audio to a temporary file and run audio analysis
    """
    if not audio_base64:
        return None

    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Run audio analysis
        print(f"  🎙️ Processing audio file: {tmp_path}")
        result = run_audio_analysis(tmp_path, verbose=False, enable_tts=False)

        # Clean up
        os.unlink(tmp_path)

        return result

    except Exception as e:
        print(f"  Audio processing error: {e}")
        return None


def process_video_frames(frames_base64: list) -> dict:
    """
    Process video frames through vision model
    """
    if not frames_base64:
        return None

    try:
        # Check if Ollama is running first
        if not ollama_running():
            print("  ⚠️ Ollama not running - cannot process video")
            return None

        # Format frames for vision_analyse
        formatted_frames = []
        for i, b64_str in enumerate(frames_base64):
            formatted_frames.append({"timestamp_s": i * 2.0, "b64": b64_str})

        # Get vision model
        models = available_models()
        vision_model = pick_vision_model(models)

        if not vision_model:
            print("  No vision model available")
            return None

        # Run vision analysis
        print(f"  🎬 Processing {len(frames_base64)} frames with {vision_model}")
        result = vision_analyse(formatted_frames, vision_model, workers=2)

        return result

    except Exception as e:
        print(f"  Video processing error: {e}")
        return None


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    # Get inputs
    frames = data.get("frames", [])
    audio_base64 = data.get("audBase64", "")
    audio_desc = data.get("audDesc", "")
    logs = data.get("logs", [])
    zone = data.get("zone", "UNKNOWN")
    context_notes = data.get("ctx", "")

    print(f"\n📥 Received analysis request:")
    print(f"   Frames: {len(frames)}")
    print(f"   Audio: {'Yes (base64)' if audio_base64 else 'No'}")
    print(f"   Audio desc: {audio_desc[:50] if audio_desc else 'None'}")
    print(f"   Logs: {len(logs)}")
    print(f"   Zone: {zone}")

    # ── 1. Process Video ──
    video_analysis = None
    if frames:
        video_analysis = process_video_frames(frames)
        if video_analysis:
            print(
                f"   ✓ Video analysis complete: {video_analysis.get('primary_event_type', '?')}"
            )

    # ── 2. Process Audio ──
    audio_analysis = None

    # Priority 1: Base64 audio from file upload
    if audio_base64:
        audio_analysis = process_audio_base64(audio_base64)
        if audio_analysis:
            print(
                f"   ✓ Audio analysis complete: {audio_analysis.get('primary_event_type', '?')}"
            )

    # Priority 2: Audio description (if no base64)
    elif audio_desc:
        # Create a simple analysis from description
        audio_analysis = {
            "primary_event_type": (
                "panic_call"
                if any(
                    kw in audio_desc.lower()
                    for kw in ["help", "scream", "emergency", "gun", "bang"]
                )
                else "normal_activity"
            ),
            "confidence": (
                70
                if any(kw in audio_desc.lower() for kw in ["gun", "bang", "explosion"])
                else 50
            ),
            "scene_summary": audio_desc[:100],
            "detected_sounds": [audio_desc[:30]],
            "has_distress_signals": any(
                kw in audio_desc.lower() for kw in ["help", "scream", "emergency"]
            ),
            "has_weapon_audio": any(
                kw in audio_desc.lower() for kw in ["gun", "bang", "shot"]
            ),
            "has_explosion": "explosion" in audio_desc.lower(),
            "has_fire_audio": "fire" in audio_desc.lower(),
            "is_panic_call": "help" in audio_desc.lower()
            or "emergency" in audio_desc.lower(),
            "threat_rationale": f"Audio description indicates {audio_desc[:50]}",
            "zone_guess": zone,
            "notes": "From text description",
            "transcript_text": audio_desc,
            "distress_keywords": [
                kw
                for kw in ["help", "scream", "emergency", "gun", "fire"]
                if kw in audio_desc.lower()
            ],
        }
        print(f"   ✓ Audio from description: {audio_analysis['primary_event_type']}")

    # ── 3. Process Logs ──
    log_signals = []
    for log in logs:
        try:
            event_type = EventType(log.get("event_type", "motion_detection"))
        except ValueError:
            event_type = EventType.MOTION_DETECTION

        log_signals.append(
            Signal(
                type=event_type,
                confidence=log.get("confidence", 50),
                timestamp=datetime.now(),
                location=log.get("location", zone),
                metadata=log.get("metadata", {}),
            )
        )

    # ── 4. Fuse Everything ──
    print(f"\n🔄 Fusing signals...")

    # Create signals from analyses
    signals, fusion_context, fusion_summary = fuse_signals(
        video_analysis, audio_analysis, zone
    )

    # Add log signals
    signals.extend(log_signals)

    if not signals:
        # Create a default signal
        signals = [
            Signal(
                type=EventType.MOTION_DETECTION,
                confidence=30,
                timestamp=datetime.now(),
                location=zone,
                metadata={"source": "default"},
            )
        ]

    # ── 5. Run Security Agent ──
    agent = SecurityAgent()
    threat_level = agent.analyze_signal_fusion(signals, fusion_context)
    avg_conf = sum(s.confidence for s in signals) / len(signals)
    rec = agent.proportionality_encoding(threat_level, avg_conf, fusion_context)
    temporal = agent.temporal_reasoning(signals, fusion_context)
    attack = agent.detect_sensor_attack(signals)

    # ── 6. Generate Briefing ──
    briefing = officer_briefing(
        video_analysis=video_analysis,
        audio_analysis=audio_analysis,
        fusion_summary=fusion_summary,
        threat=threat_level,
        action=rec.action,
        visual_cue=rec.visual_cue,
        temporal=temporal,
        tracking=None,
    )

    # ── 7. Prepare Response ──
    result = {
        "threat_level": threat_level.value,
        "confidence": rec.confidence.confidence,
        "alternative_scenario": rec.confidence.alternative,
        "alternative_confidence": rec.confidence.alternative_confidence,
        "action": rec.action,
        "urgency": rec.urgency,
        "requires_human_approval": rec.requires_human_approval,
        "visual_cue": rec.visual_cue,
        "sequence_detected": temporal["sequence_detected"],
        "temporal_window_minutes": temporal["window_minutes"],
        "sensor_attack_alert": attack,
        "reasoning": rec.reasoning,
        "officer_briefing": briefing,
        "fusion_rules_applied": fusion_summary.get("fusion_rules_applied", []),
    }

    print(f"\n✅ Analysis complete: {threat_level.value.upper()} - {rec.action}")
    print("\n" + "=" * 50)
    print("📤 SERVER SENDING TO HTML:")
    print(f"   threat_level: {result['threat_level']}")
    print(f"   action: {result['action']}")
    print(f"   visual_cue: {result['visual_cue']}")
    print(f"   officer_briefing: {result['officer_briefing'][:100]}...")
    print("=" * 50)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    try:
        from video_analysis_pipeline import ollama_running

        ollama_status = "running" if ollama_running() else "stopped"
    except Exception as e:
        print(f"Health check error: {e}")
        ollama_status = "unknown"
    return jsonify({"status": "ok", "ollama": ollama_status})


# THIS IS THE CRITICAL PART - Make sure app is defined before running!
if __name__ == "__main__":
    print("🚀 CERTIS Security Server running on http://localhost:5000")
    print("   Make sure Ollama is running with: ollama serve")
    print("   Vision models required: ollama pull llava")
    print("   Audio models: pip install librosa openai-whisper soundfile")
    app.run(port=5000, debug=False)
