"use client";

import Draggable from 'react-draggable';
import React from 'react';

const DraggableSquare = ({ id, name }) => {
  return (
    <Draggable>
      <div className="w-12 h-12 bg-blue-500 m-2 flex items-center justify-center rounded">
        {name}
      </div>
    </Draggable>
  );
};


export default DraggableSquare;
