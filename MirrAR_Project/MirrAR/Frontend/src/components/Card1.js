import React from 'react';
import { Link } from "react-router-dom";

function Card() {
  return (
    <>
    <div className="card" style={{width: "18rem"}}>
    <img style={{height: "12rem"}} src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpasravXlfqdAK4nf6HKwexKaimPtbpTgx0A&usqp=CAU" className="card-img-top" alt="..."/>
    <div className="card-body">
        <h5 className="card-title">T-Shirts</h5>
        <p className="card-text">Choose your favourite t-shirt from our collection and try it on.</p>
        <Link to="/tshirt"><a href="#" className="btn btn-primary">Take me</a></Link>
    </div>
    </div>


    </>

  );
}

export default Card;
