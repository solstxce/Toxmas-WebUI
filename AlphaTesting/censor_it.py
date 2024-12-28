import os
import logging
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
from nudenet import NudeDetector
import moviepy.config as cfg

cfg.change_settings({"IMAGEMAGICK_BINARY": "magick.exe"})

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.detector = NudeDetector()

    def process_video(self, input_path):
        logger.info(f"Processing video: {input_path}")
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            out_path = self._get_output_path(input_path, 'censored_video.mp4')
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            self._process_frames(cap, out, total_frames)
            
            cap.release()
            out.release()
            
            self._add_watermark(out_path, input_path)
            
            logger.info(f"Video processing complete. Output saved to: {out_path}")
            return out_path
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals():
                out.release()

    def _process_frames(self, cap, out, total_frames):
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            logger.debug(f"Processing frame {frame_count}/{total_frames}")
            try:
                # Detect nude regions
                detections = self.detector.detect(frame)
                
                # Apply intense blur to detected regions
                for detection in detections:
                    box = detection['box']
                    x1, y1, x2, y2 = map(int, box)
                    frame = self._apply_intense_blur(frame, x1, y1, x2, y2)
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

    def _apply_intense_blur(self, image, x1, y1, x2, y2):
        # Extract the region to be blurred
        region = image[y1:y2, x1:x2]
        
        # Apply an intense blur
        blurred = cv2.GaussianBlur(region, (99, 99), 30)
        
        # Replace the original region with the blurred version
        image[y1:y2, x1:x2] = blurred
        
        return image

    def _add_watermark(self, video_path, input_path):
        logger.info("Adding watermark to video")
        video = VideoFileClip(video_path)
        watermark = (TextClip("Verified by NoHate", fontsize=24, color='white', font='Arial')
                     .set_position(('right', 'bottom'))
                     .set_duration(video.duration))
        
        final_video = CompositeVideoClip([video, watermark])
        final_output_path = self._get_output_path(input_path, 'final_censored_video.mp4')
        final_video.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

    def _get_output_path(self, input_path, filename):
        return os.path.join(os.path.dirname(input_path), filename)

def test_video_censoring(input_video_path):
    logger.info(f"Starting video censoring test for: {input_video_path}")
    
    try:
        processor = VideoProcessor()
        censored_video_path = processor.process_video(input_video_path)
        
        if os.path.exists(censored_video_path):
            logger.info(f"Video censoring successful. Censored video saved at: {censored_video_path}")
            
            original_size = os.path.getsize(input_video_path)
            censored_size = os.path.getsize(censored_video_path)
            
            logger.info(f"Original video size: {original_size / 1024 / 1024:.2f} MB")
            logger.info(f"Censored video size: {censored_size / 1024 / 1024:.2f} MB")
            
            size_diff_percent = (censored_size - original_size) / original_size * 100
            logger.info(f"Size difference: {size_diff_percent:.2f}%")
            
            return True
        else:
            logger.error("Censored video was not created.")
            return False
    
    except Exception as e:
        logger.error(f"An error occurred during video censoring: {str(e)}")
        return False

def main():
    test_video_path = r"C:\Users\Solstxce\Projects\CSP_Project_NoHate\Document-Analyzer V1\input_test_Data.mp4"
    success = test_video_censoring(test_video_path)
    
    if success:
        logger.info("Video censoring test completed successfully.")
    else:
        logger.info("Video censoring test failed.")

if __name__ == "__main__":
    main()