#!/bin/bash
gz service -s /world/default/set_pose   --reqtype gz.msgs.Pose   --reptype gz.msgs.Boolean   --req 'name: "kyon"
        position { x: 0.0, y: 0.0, z: 1.0 } 
        orientation { w: 1.0 }'