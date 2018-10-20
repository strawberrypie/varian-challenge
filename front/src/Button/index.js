import React from 'react';

export default class Button extends React.Component {

    renderButton = ({ text, onClick }) =>
        <button className="button" onClick = { onClick }>
            {text}
        </button>

    render = () => this.renderButton( this.props )

}