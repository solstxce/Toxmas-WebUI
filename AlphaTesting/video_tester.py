import os
import logging
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip
from nudenet import NudeDetector
from PIL import Image
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
            
            if frame_count % 5 == 0:  # Process every 5th frame for performance
                logger.debug(f"Processing frame {frame_count}/{total_frames}")
                try:
                    # Convert frame to PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # Save frame as temporary file
                    temp_frame_path = f'temp_frame_{frame_count}.jpg'
                    pil_image.save(temp_frame_path)
                    
                    # Process the frame
                    censored_img_path = self.detector.censor(temp_frame_path)
                    if censored_img_path:
                        # Read the censored image and convert back to OpenCV format
                        censored_frame = cv2.cvtColor(cv2.imread(censored_img_path), cv2.COLOR_BGR2RGB)
                        frame = censored_frame
                    else:
                        logger.warning(f"Censorship returned None for frame {frame_count}")
                    
                    # Clean up temporary files
                    os.remove(temp_frame_path)
                    if censored_img_path and os.path.exists(censored_img_path):
                        os.remove(censored_img_path)
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

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

    def analyze_image(self, image_path):
        logger.info(f"Starting image analysis for: {image_path}")
        
        try:
            censored_img_path = self.detector.censor(image_path)
            if censored_img_path:
                censored_img = Image.open(censored_img_path)
                original_img = Image.open(image_path)
                
                diff = np.array(original_img) - np.array(censored_img)
                overall_score = np.sum(np.abs(diff)) / (original_img.size[0] * original_img.size[1] * 3)
                
                stats = {
                    'overall_score': overall_score,
                    'censored_image_path': censored_img_path
                }
                
                logger.info(f"Image analysis complete. Overall score: {overall_score}")
                logger.debug(f"Detailed stats: {stats}")
                
                return stats
            else:
                logger.warning(f"Censorship returned None for {image_path}")
                return None
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return None

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
    test_video_path = r"C:\Users\Solstxce\Projects\CSP_Project_NoHate\Document-Analyzer V1\Anushka.mp4"
    success = test_video_censoring(test_video_path)
    
    if success:
        logger.info("Video censoring test completed successfully.")
    else:
        logger.info("Video censoring test failed.")

if __name__ == "__main__":
    main()