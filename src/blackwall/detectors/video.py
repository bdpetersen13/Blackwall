"""
Video Detection Implementation for Blackwall
Detects AI-Generated Videos by Analyzing Video Frames
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import tempfile
import shutil
import cv2
import ffmpeg
from blackwall.detectors.base import BaseDetector
from blackwall.detectors import register_detector
from blackwall.detectors.image import ImageDetector
from blackwall.utils.logger import get_logger
from blackwall.utils.exceptions import InvalidInputError, ProcessingTimeoutError


logger = get_logger(__name__)


class VideoDetector(BaseDetector):
    """ Detector for AI-Generated Videos """
    
    def __init__(self):
        super().__init__()
        self.image_detector = ImageDetector()
        self.model_version = "video_v1.0"
        self.frames_to_sample = 10
        self.max_duration_seconds = 300  # 5 minutes
    
    @property
    def detector_type(self) -> str:
        return "video"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"]
    
    @property
    def model_path(self) -> Path:
        # Video detector uses image detector's model
        return self.config.image_model_path
    
    def load_model(self) -> None:
        """ Load the Video Detection Model (Uses Image Detection) """
        self.logger.info("loading_video_model")
        
        # Load image detector model
        self.image_detector.load_model()
        
        self._model_loaded = True
        self.logger.info("video_model_loaded")
    
    def preprocess(self, file_path: Path) -> Dict[str, Any]:
        """ Preprocess Video File for Detection """
        try:
            # Get video info
            video_info = self._get_video_info(file_path)
            
            # Validate video
            self._validate_video(video_info)
            
            # Extract frames
            frames_dir = self._extract_frames(file_path, video_info)
            
            return {
                "frames_dir": frames_dir,
                "video_info": video_info,
                "frame_count": len(list(frames_dir.glob("*.jpg")))
            }
            
        except Exception as e:
            self.logger.error(
                "video_preprocessing_failed",
                file_path = str(file_path),
                error = str(e)
            )
            raise
    
    def run_inference(
        self,
        preprocessed_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """ Run Inference on Extracted Video Frames """
        try:
            frames_dir = preprocessed_data["frames_dir"]
            video_info = preprocessed_data["video_info"]
            
            # Analyze each frame
            frame_results = []
            frame_paths = sorted(frames_dir.glob("*.jpg"))
            
            for frame_path in frame_paths:
                try:
                    # Run image detection on frame
                    result = self.image_detector.detect(frame_path)
                    frame_results.append({
                        "frame": frame_path.stem,
                        "probability": result.probability,
                        "metadata": result.metadata
                    })
                except Exception as e:
                    self.logger.warning(
                        "frame_analysis_failed",
                        frame = str(frame_path),
                        error = str(e)
                    )
            
            # Clean up frames directory
            shutil.rmtree(frames_dir)
            
            if not frame_results:
                raise ModelInferenceError("No frames could be analyzed")
            
            # Aggregate results
            probabilities = [r["probability"] for r in frame_results]
            
            # Use various aggregation methods
            mean_probability = np.mean(probabilities)
            median_probability = np.median(probabilities)
            max_probability = np.max(probabilities)
            
            # Weight towards higher probabilities (if any frame is AI, video likely is)
            final_probability = 0.5 * mean_probability + 0.3 * max_probability + 0.2 * median_probability
            
            # Analyze temporal consistency
            temporal_score = self._analyze_temporal_consistency(probabilities)
            
            # Adjust based on temporal analysis
            if temporal_score > 0.8:
                final_probability = min(final_probability * 1.1, 1.0)
            
            metadata = {
                "frame_count": len(frame_results),
                "mean_probability": round(mean_probability, 4),
                "median_probability": round(median_probability, 4),
                "max_probability": round(max_probability, 4),
                "temporal_consistency": round(temporal_score, 4),
                "video_duration": video_info["duration"],
                "video_resolution": f"{video_info['width']}x{video_info['height']}",
                "fps": video_info["fps"],
                "codec": video_info["codec"],
                "ai_frame_count": sum(1 for p in probabilities if p > 0.5),
                "frame_analysis": self._summarize_frame_results(frame_results)
            }
            
            return final_probability, metadata
            
        except Exception as e:
            self.logger.error(
                "video_inference_failed",
                error = str(e),
                exc_info = True
            )
            raise
    
    def _get_video_info(self, file_path: Path) -> Dict[str, Any]:
        """ Get Video File Information using ffmpeg."""
        try:
            probe = ffmpeg.probe(str(file_path))
            
            # Extract video stream info
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise InvalidInputError("No video stream found in file")
            
            # Extract relevant info
            return {
                "duration": float(probe['format'].get('duration', 0)),
                "width": int(video_stream.get('width', 0)),
                "height": int(video_stream.get('height', 0)),
                "fps": eval(video_stream.get('r_frame_rate', '0/1')),
                "codec": video_stream.get('codec_name', 'unknown'),
                "bit_rate": int(probe['format'].get('bit_rate', 0)),
                "nb_frames": int(video_stream.get('nb_frames', 0))
            }
            
        except Exception as e:
            raise InvalidInputError(f"Failed to probe video: {str(e)}")
    
    def _validate_video(self, video_info: Dict[str, Any]) -> None:
        """ Validate Video for Processing """
        # Check duration
        if video_info["duration"] > self.max_duration_seconds:
            raise InvalidInputError(
                f"Video too long: {video_info['duration']}s > "
                f"{self.max_duration_seconds}s"
            )
        
        # Check resolution
        if video_info["width"] < 64 or video_info["height"] < 64:
            raise InvalidInputError("Video resolution too low")
        
        # Check if video has frames
        if video_info["duration"] == 0 and video_info["nb_frames"] == 0:
            raise InvalidInputError("Video appears to be empty")
    
    def _extract_frames(
        self,
        file_path: Path,
        video_info: Dict[str, Any]
    ) -> Path:
        """ Extract Frames from Video """
        # Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp(prefix="blackwall_frames_"))
        
        try:
            duration = video_info["duration"]
            
            # Calculate frame extraction interval
            if duration > 0:
                interval = duration / self.frames_to_sample
            else:
                # If duration is unknown, extract every N frames
                interval = video_info["nb_frames"] / self.frames_to_sample
            
            self.logger.info(
                "extracting_frames",
                video_path = str(file_path),
                frames = self.frames_to_sample,
                interval = interval
            )
            
            # Use ffmpeg to extract frames
            (
                ffmpeg
                .input(str(file_path))
                .filter('fps', fps = 1/interval)
                .output(
                    str(temp_dir / 'frame_%04d.jpg'),
                    vframes = self.frames_to_sample,
                    **{'qscale:v': 2}  # High quality JPEG
                )
                .overwrite_output()
                .run(quiet = True)
            )
            
            # Verify frames were extracted
            frame_count = len(list(temp_dir.glob("*.jpg")))
            if frame_count == 0:
                raise InvalidInputError("Failed to extract any frames from video")
            
            self.logger.info(
                "frames_extracted",
                count = frame_count,
                directory = str(temp_dir)
            )
            
            return temp_dir
            
        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise InvalidInputError(f"Frame extraction failed: {str(e)}")
    
    def _analyze_temporal_consistency(self, probabilities: List[float]) -> float:
        """ Analyze Temporal Consistency of AI Detection Across Frames """
        if len(probabilities) < 2:
            return 1.0
        
        # Calculate variance in probabilities
        variance = np.var(probabilities)
        
        # Low variance means consistent detection (likely all AI or all real)
        # High variance means mixed detection (suspicious)
        consistency_score = 1.0 - min(variance * 4, 1.0)
        
        return consistency_score
    
    def _summarize_frame_results(
        self,
        frame_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ Summarize frame Analysis Results """
        # Group frames by probability ranges
        ranges = {
            "very_high": [r for r in frame_results if r["probability"] > 0.9],
            "high": [r for r in frame_results if 0.7 < r["probability"] <= 0.9],
            "medium": [r for r in frame_results if 0.3 < r["probability"] <= 0.7],
            "low": [r for r in frame_results if r["probability"] <= 0.3]
        }
        
        return {
            "distribution": {
                k: len(v) for k, v in ranges.items()
            },
            "most_ai_frame": max(frame_results, key=lambda x: x["probability"])["frame"],
            "least_ai_frame": min(frame_results, key=lambda x: x["probability"])["frame"]
        }


# Register the detector
register_detector("video", VideoDetector)