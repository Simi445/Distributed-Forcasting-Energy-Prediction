import { CardSmall } from './MLCard';

export default function IdleComponent({changeState, models})
  {
    return(
      <>
      <h1 className="text-3xl font-bold mb-4 text-center">Model Section</h1>
      <div className="flex justify-center gap-6 flex-wrap">
        {models.map((model) => (
          <div key={model.id} className="min-w-[12rem] flex-1">
             <CardSmall cardTitle={model.cardTitle} onClick={() => changeState(model.cardTitle)}/>
          </div>
        ))}
      </div>
      </>
    );
  }