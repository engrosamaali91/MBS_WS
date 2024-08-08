#!/usr/bin/env python

import rospy
import sys
import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose

"""
Publish a frame 3D pose as a PoseStamped continuously.

Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
"""


class PublishFrameAsPoseStamped(object):
    def __init__(self, frame_to_posestamped,
                 reference_frame,
                 rate,
                 verbose=False):
        """
        Class to publish a frame as a PoseStamped.
        :param frame_to_posestamped str: frame that will be published its
                pose as PoseStamped.
        :param reference_frame str: frame that will be the header.frame_id
                of the PoseStamped.
        :param rate int: rate at which to compute and publish the pose.
        :param verbose bool: print to screen the transformations.
        """
        self.buffer = Buffer(rospy.Duration(5.0))
        self.tf_l = TransformListener(self.buffer)
        topic_name = frame_to_posestamped.replace('/', '')
        self.pose_pub = rospy.Publisher(topic_name + '_as_posestamped',
                                        PoseStamped, queue_size=1)
        self.frame_to_posestamped = frame_to_posestamped
        self.reference_frame = reference_frame
        self.rate = rospy.Rate(rate)
        self.verbose = verbose

    def _transform_tf2(self, pose, from_frame, to_frame,
                       wait_duration=1.0):
        ps = PoseStamped()
        ps.pose = pose
        ps.header.frame_id = from_frame
        # try:
        transform = self.buffer.lookup_transform(to_frame,
                                                 ps.header.frame_id,
                                                 rospy.Time(0),
                                                 rospy.Duration(wait_duration))

        print(f'Hello world:{ps}')
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        #     rospy.logwarn("Exception transforming: " + str(e))
        return do_transform_pose(ps, transform)

    def transform_pose(self, pose, from_frame, to_frame):
        """
        Transform the 'pose' from frame 'from_frame'
         to frame 'to_frame'

        :param geometry_msgs/Pose pose: 3D Pose to transform.
        :param str from_frame: frame that the pose belongs to.
        :param str to_frame: to what frame transform.
        """
        ps = PoseStamped()
        # ps.header.stamp = #self.tf_l.getLatestCommonTime(from_frame,
        # to_frame)
        ps.header.frame_id = from_frame
        ps.pose = pose
        transform_ok = False
        min_time_in_between_warns = rospy.Duration(5.0)
        last_warn = rospy.Time.now() - min_time_in_between_warns
        while not transform_ok and not rospy.is_shutdown():
            try:
                target_ps = self._transform_tf2(pose, from_frame, to_frame)
                # target_ps = self.tf_l.transformPose(to_frame, ps)
                transform_ok = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                if rospy.Time.now() > (last_warn + min_time_in_between_warns):
                    rospy.logwarn(
                        "Exception on transforming pose... trying again \n(" +
                        str(e) + ")")
                    last_warn = rospy.Time.now()
                rospy.sleep(0.2)
                ps.header.stamp = self.tf_l.getLatestCommonTime(
                    from_frame, to_frame)

        target_ps.header.stamp = rospy.Time.now()
        return target_ps

    def run(self):
        ps = Pose()
        ps.orientation.w = 1.0  # Quaternion must be correct
        while not rospy.is_shutdown():
            # We transform a pose with reference frame
            # self.frame_to_posestamped
            # which is 0.0, 0.0, 0.0
            # to the reference frame to get it's pose
            tfed_ps = self.transform_pose(ps,
                                          self.frame_to_posestamped,
                                          self.reference_frame)
            self.pose_pub.publish(tfed_ps)
            if self.verbose:
                print(tfed_ps)
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('frame_to_posestamped')
    argv = rospy.myargv(sys.argv)
    if len(argv) < 3:
        print("Usage:")
        print(argv[0] + " frame_to_posestamped reference_frame [rate]")
        exit(0)
    frame_to_posestamped = argv[1]
    reference_frame = argv[2]
    if len(argv) == 4:
        rate = int(argv[3])
    else:
        rate = 10
    pfaps = PublishFrameAsPoseStamped(frame_to_posestamped,
                                      reference_frame,
                                      rate,
                                      verbose=False)
    pfaps.run()