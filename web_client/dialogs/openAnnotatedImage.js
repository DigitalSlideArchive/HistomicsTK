import $ from 'jquery';
import _ from 'underscore';

import { restRequest } from 'girder/rest';
import ItemCollection from 'girder/collections/ItemCollection';
import UserCollection from 'girder/collections/UserCollection';
import View from 'girder/views/View';
import 'girder/utilities/jquery/girderModal';

import events from '../events';
import router from '../router';

import listTemplate from '../templates/dialogs/annotatedImageList.pug';
import template from '../templates/dialogs/openAnnotatedImage.pug';
import '../stylesheets/dialogs/openAnnotatedImage.styl';

let dialog;
const paths = {};

const AnnotatedImageList = View.extend({
    events: {
        'keyup .form-control': 'fetch'
    },

    initialize() {
        this.listenTo(this.collection, 'reset', this.render);
    },

    render() {
        this.$el.html(listTemplate({
            items: this.collection.toJSON(),
            paths
        }));
        return this;
    }
});

const OpenAnnotatedImage = View.extend({
    events: {
        'click .h-annotated-image': '_submit',
        'keyup input': '_debouncedFetch',
        'change select': '_debouncedFetch'
    },

    initialize() {
        this.collection = new ItemCollection();
        // disable automatic sorting of this collection
        this.collection.comparator = null;

        this._users = new UserCollection();
        this._users.sortField = 'login';
        this._users.pageLimit = 500;
        this._usersIsFetched = false;
        this._users.fetch().done(() => {
            this._usersIsFetched = true;
            this.render();
        });

        if (createDialog.debounceTimeout > 0) {
            this._debouncedFetch = _.debounce(this.fetch, createDialog.debounceTimeout);
        } else {
            this._debouncedFetch = this.fetch;
        }
    },

    render() {
        if (!this._usersIsFetched) {
            return this;
        }
        this.$el.html(template({
            imageName: this._imageName,
            creator: this._creator,
            users: this._users
        })).girderModal(this);
        this.$el.tooltip();

        new AnnotatedImageList({
            parentView: this,
            collection: this.collection,
            el: this.$('.h-annotated-images-list-container')
        }).render();
        return this;
    },

    fetch() {
        const data = {
            limit: 10
        };
        let items;
        let changed = false;

        const creator = this.$('#h-annotation-creator').val();
        if (this._creator !== creator) {
            this._creator = creator;
            changed = true;

            if (creator) {
                data.creatorId = creator;
            }
        }

        const imageName = (this.$('#h-image-name').val() || '').trim();
        if (this._imageName !== imageName) {
            this._imageName = imageName;
            changed = true;

            if (imageName) {
                data.imageName = this._imageName;
            }
        }

        if (!changed) {
            return $.Deferred().resolve(this.collection).promise();
        }

        return restRequest({
            url: 'annotation/images',
            data
        }).then((_items) => {
            items = _items;
            const promises = _.map(items, (item) => {
                return this._getResourcePath(item);
            });
            return $.when(...promises);
        }).then(() => {
            this.collection.reset(items);
            return this.collection;
        });
    },

    _submit(evt) {
        const id = this.$(evt.currentTarget).data('id');
        router.setQuery('bounds', null, {trigger: false});
        router.setQuery('image', id, {trigger: true});
        this.$el.modal('hide');
    },

    _getResourcePath(item) {
        if (_.has(paths, item._id)) {
            return $.Deferred().resolve(paths[item._id]).promise();
        }

        return restRequest({
            url: `resource/${item._id}/path`,
            data: {
                type: 'item'
            }
        }).done((path) => {
            paths[item._id] = path;
        });
    }
});

function createDialog() {
    return new OpenAnnotatedImage({
        parentView: null
    });
}

createDialog.debounceTimeout = 500;

events.on('h:openAnnotatedImageUi', function () {
    if (!dialog) {
        dialog = createDialog();
    }
    dialog.setElement($('#g-dialog-container')).render().fetch();
});

export default createDialog;
