import { useState, useEffect, useContext } from 'react';
import { CardSmall } from './MLCard';

export default function IdleComponent({changeState, models})
  {
    return(
      <>
      <h1 className="text-3xl font-bold mb-4 ">Model Section</h1>
      <div className="grid gap-6 md:grid-cols-1 ">
        {models.map((model) => (
          <CardSmall onClick = {()=> changeState(model.label)}/>
        ))}
      </div>
      </>
    );
  }