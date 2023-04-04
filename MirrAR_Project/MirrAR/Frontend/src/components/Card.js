import React from 'react';
import { Link } from "react-router-dom";

function Card() {
  return (
    <>
    <div className="card" style={{width: "18rem"}}>
    <img style={{height: "12rem"}} src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTndJbq0-rKWzIoi8AREsUKfDhfa2jMBkLqf212imfr1kWBOMHVawEGAsSD8SqeZVYdvZQ&usqp=CAU" className="card-img-top" alt="..."/>
    <div className="card-body">
        <h5 className="card-title">Lipsticks</h5>
        <p className="card-text">Choose your favourite lipstick from our collection and try it on.</p>
        <Link to="/lipproducts"><a href="#" className="btn btn-primary">Take me</a></Link>
    </div>
    </div>


    </>

  );
}

export default Card;
