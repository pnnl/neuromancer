import React, { useState, useEffect } from 'react';
import DraggableSquare from '@/components/ui/draggablesquare'; // Import DraggableSquare

export default function EnvironmentPane({ environmentSquares, setEnvironmentSquares }) {
    // Function to update local storage with environment squares
    const updateLocalStorage = (squares) => {
      localStorage.setItem('environmentSquares', JSON.stringify(squares));
    };
  
    // Effect to update local storage whenever environment squares change
    useEffect(() => {
      updateLocalStorage(environmentSquares);
    }, [environmentSquares]);
  
    return (
      <div className="fixed inset-y-0 right-0 flex flex-col w-64 border-l border-gray-200 dark:border-gray-800">
        {environmentSquares.map(square => (
          <DraggableSquare key={square.id} name={square.name} position={square.position} />
        ))}
      </div>
    );
  }
  
  function OptionIcon(props) {
    return (
      <svg
        {...props}
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M3 3h6l6 18h6" />
        <path d="M14 3h7" />
      </svg>
    )
  }
  
  
  function PhoneIcon(props) {
    return (
      <svg
        {...props}
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
      </svg>
    )
  }
  
  
  function VariableIcon(props) {
    return (
      <svg
        {...props}
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M8 21s-4-3-4-9 4-9 4-9" />
        <path d="M16 3s4 3 4 9-4 9-4 9" />
        <line x1="15" x2="9" y1="9" y2="15" />
        <line x1="9" x2="15" y1="9" y2="15" />
      </svg>
    )
  }