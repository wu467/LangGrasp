import pyrealsense2 as rs
import numpy as np
import cv2


def picture():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config object to configure the pipeline
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)
    align = rs.align(rs.stream.color)  # Create align object for depth-color alignment

    try:
        while True:
            # Wait for a coherent pair of frames: color and depth
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            if not aligned_frames:
                continue  # If alignment fails, go back to the beginning of the loop

            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not aligned_depth_frame:
                continue

            # Convert aligned_depth_frame and color_frame to numpy arrays
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Display the aligned depth image
            aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03),
                                                       cv2.COLORMAP_JET)
            cv2.imshow("Aligned Depth colormap", aligned_depth_colormap)

            cv2.imwrite('../../data/depthImg/test_depth.png', aligned_depth_image)

            # Display the color image
            cv2.imshow("Color Image", color_image)
            cv2.imwrite('../../data/input/test_color.png', color_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline and close all windows
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    picture()